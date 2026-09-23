#include "vn97/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace vn97 {
namespace {

constexpr std::array<std::uint8_t, 8> kMagicV1 = {'V','N','9','7','R','U','N','1'};
constexpr std::array<std::uint8_t, 8> kMagicV2 = {'V','N','9','7','R','U','N','2'};
constexpr std::uint16_t kVersionV1 = 1;
constexpr std::uint16_t kVersionV2 = 2;
constexpr std::uint16_t kHeaderSizeV1 = 64;
constexpr std::uint16_t kHeaderSizeV2 = 100;
constexpr std::uint32_t kCheckpointModelBound = 1u << 0;
constexpr std::size_t kMaxStateBytes = 512ull * 1024ull * 1024ull;

bool MulOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr) return true;
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return true;
    *out = a * b;
    return false;
}

RuntimeStatus StateCount(const RuntimeConfig& config, std::size_t* out) {
    if (out == nullptr) return RuntimeStatus::kNullArgument;
    if (config.layers == 0 || config.batch == 0 || config.d_model == 0 || config.d_state == 0) {
        return RuntimeStatus::kInvalidConfig;
    }
    std::size_t count = config.layers;
    for (std::size_t value : {static_cast<std::size_t>(config.batch),
                              static_cast<std::size_t>(config.d_model),
                              static_cast<std::size_t>(config.d_state)}) {
        if (MulOverflow(count, value, &count)) return RuntimeStatus::kSizeOverflow;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return RuntimeStatus::kSizeOverflow;
    }
    if (count * sizeof(float) > kMaxStateBytes) {
        return RuntimeStatus::kSizeOverflow;
    }
    *out = count;
    return RuntimeStatus::kOk;
}

bool ValidRecurrentBackend(RecurrentBackend backend) {
    return backend == RecurrentBackend::kAuto || backend == RecurrentBackend::kScalar ||
           backend == RecurrentBackend::kArm64Neon;
}

bool ValidPackedBackend(PackedTernaryBackend backend) {
    return backend == PackedTernaryBackend::kAuto || backend == PackedTernaryBackend::kScalar ||
           backend == PackedTernaryBackend::kArm64Neon;
}

bool SameModelId(
    const std::array<std::uint8_t, 32>& bound,
    const std::uint8_t candidate[32]) {
    return std::equal(bound.begin(), bound.end(), candidate);
}

bool StateAllZero(const std::vector<float>& state) {
    return std::all_of(
        state.begin(),
        state.end(),
        [](float value) { return value == 0.0f; });
}

RuntimeStatus MapLanguageStatus(LanguageStatus status) {
    switch (status) {
        case LanguageStatus::kOk:
            return RuntimeStatus::kOk;
        case LanguageStatus::kNullArgument:
            return RuntimeStatus::kNullArgument;
        case LanguageStatus::kOutputTooSmall:
            return RuntimeStatus::kOutputTooSmall;
        case LanguageStatus::kSizeOverflow:
            return RuntimeStatus::kSizeOverflow;
        case LanguageStatus::kBackendUnavailable:
            return RuntimeStatus::kBackendUnavailable;
        case LanguageStatus::kInvalidConfig:
        case LanguageStatus::kInvalidModel:
        case LanguageStatus::kInvalidToken:
        case LanguageStatus::kNonFinite:
            return RuntimeStatus::kInferenceError;
    }
    return RuntimeStatus::kInferenceError;
}

void WriteU16(std::uint8_t* out, std::uint16_t value) {
    out[0] = static_cast<std::uint8_t>(value & 0xffu);
    out[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
}
void WriteU32(std::uint8_t* out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu);
}
void WriteU64(std::uint8_t* out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) out[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu);
}
std::uint16_t ReadU16(const std::uint8_t* in) {
    return static_cast<std::uint16_t>(in[0]) | (static_cast<std::uint16_t>(in[1]) << 8);
}
std::uint32_t ReadU32(const std::uint8_t* in) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(in[i]) << (8 * i);
    return value;
}
std::uint64_t ReadU64(const std::uint8_t* in) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(in[i]) << (8 * i);
    return value;
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

RuntimeStatus ResolveBackends(
    const RuntimeConfig& config,
    RecurrentBackend* recurrent,
    PackedTernaryBackend* packed) {
    if (!ValidRecurrentBackend(config.recurrent_backend) || !ValidPackedBackend(config.packed_backend)) {
        return RuntimeStatus::kInvalidConfig;
    }
    const auto rr = ResolveRecurrentBackend(config.recurrent_backend);
    const auto rp = ResolvePackedTernaryBackend(config.packed_backend);
    if (!RecurrentBackendAvailable(rr) || !PackedTernaryBackendAvailable(rp)) {
        return RuntimeStatus::kBackendUnavailable;
    }
    *recurrent = rr;
    *packed = rp;
    return RuntimeStatus::kOk;
}

}  // namespace

RuntimeStatus RuntimeSession::Create(const RuntimeConfig& config, RuntimeSession** out) {
    if (out == nullptr) return RuntimeStatus::kNullArgument;
    *out = nullptr;
    std::size_t count = 0;
    RuntimeStatus status = StateCount(config, &count);
    if (status != RuntimeStatus::kOk) return status;
    RecurrentBackend recurrent;
    PackedTernaryBackend packed;
    status = ResolveBackends(config, &recurrent, &packed);
    if (status != RuntimeStatus::kOk) return status;
    std::unique_ptr<RuntimeSession> session(new RuntimeSession());
    try {
        session->state_.assign(count, 0.0f);
    } catch (...) {
        return RuntimeStatus::kSizeOverflow;
    }
    session->config_ = config;
    session->resolved_recurrent_backend_ = recurrent;
    session->resolved_packed_backend_ = packed;
    *out = session.release();
    return RuntimeStatus::kOk;
}

RuntimeStatus RuntimeSession::Activate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kCreated) return RuntimeStatus::kInvalidLifecycle;
    lifecycle_ = RuntimeLifecycle::kActive;
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::Suspend() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kActive) return RuntimeStatus::kInvalidLifecycle;
    lifecycle_ = RuntimeLifecycle::kSuspended;
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::Resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kSuspended) return RuntimeStatus::kInvalidLifecycle;
    lifecycle_ = RuntimeLifecycle::kActive;
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::Advance(std::uint64_t token_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kActive) return RuntimeStatus::kInvalidLifecycle;
    if (model_bound_) return RuntimeStatus::kInferenceError;
    if (token_count > std::numeric_limits<std::uint64_t>::max() - sequence_position_) {
        return RuntimeStatus::kCounterOverflow;
    }
    sequence_position_ += token_count;
    return RuntimeStatus::kOk;
}

RuntimeStatus RuntimeSession::InferStep(
    const LanguageModelView& model,
    const std::uint32_t* input_ids,
    float* logits,
    std::size_t logits_count) {
    return InferStepOutput(model, input_ids, logits, logits_count, false);
}

RuntimeStatus RuntimeSession::InferStepHidden(
    const LanguageModelView& model,
    const std::uint32_t* input_ids,
    float* hidden,
    std::size_t hidden_count) {
    return InferStepOutput(model, input_ids, hidden, hidden_count, true);
}

RuntimeStatus RuntimeSession::InferStepOutput(
    const LanguageModelView& model,
    const std::uint32_t* input_ids,
    float* logits,
    std::size_t logits_count,
    bool hidden_only) {
    if (input_ids == nullptr || logits == nullptr) return RuntimeStatus::kNullArgument;
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kActive) return RuntimeStatus::kInvalidLifecycle;
    if (model.n_layers != config_.layers ||
        model.d_model != config_.d_model ||
        model.d_state != config_.d_state) {
        return RuntimeStatus::kModelMismatch;
    }
    if (model_bound_) {
        if (!SameModelId(model_id_, model.model_id)) {
            return RuntimeStatus::kModelMismatch;
        }
    } else {
        if (sequence_position_ != 0 || !StateAllZero(state_)) {
            return RuntimeStatus::kModelMismatch;
        }
        const auto model_status = ValidateLanguageModel(model);
        if (model_status != LanguageStatus::kOk) {
            return MapLanguageStatus(model_status);
        }
    }
    if (sequence_position_ == std::numeric_limits<std::uint64_t>::max()) {
        return RuntimeStatus::kCounterOverflow;
    }

    std::size_t workspace_count = 0;
    const auto workspace_status =
        LanguageWorkspaceFloats(model, config_.batch, &workspace_count);
    if (workspace_status != LanguageStatus::kOk) {
        return MapLanguageStatus(workspace_status);
    }
    try {
        if (language_workspace_.size() != workspace_count) {
            language_workspace_.assign(workspace_count, 0.0f);
        }
    } catch (...) {
        return RuntimeStatus::kSizeOverflow;
    }

    const auto status = hidden_only
        ? LanguageStepHiddenF32WithBackends(
            model,
            input_ids,
            config_.batch,
            state_.data(),
            logits,
            logits_count,
            language_workspace_.data(),
            language_workspace_.size(),
            resolved_recurrent_backend_,
            resolved_packed_backend_)
        : LanguageStepF32WithBackends(
            model,
            input_ids,
            config_.batch,
            state_.data(),
            logits,
            logits_count,
            language_workspace_.data(),
            language_workspace_.size(),
            resolved_recurrent_backend_,
            resolved_packed_backend_);
    if (status != LanguageStatus::kOk) {
        return MapLanguageStatus(status);
    }

    if (!model_bound_) {
        std::copy(
            model.model_id,
            model.model_id + 32,
            model_id_.begin());
        model_bound_ = true;
    }
    ++sequence_position_;
    return RuntimeStatus::kOk;
}

RuntimeStatus RuntimeSession::ModelBinding(
    bool* bound,
    std::uint8_t* model_id,
    std::size_t model_id_capacity) const {
    if (bound == nullptr || model_id == nullptr) return RuntimeStatus::kNullArgument;
    if (model_id_capacity < model_id_.size()) return RuntimeStatus::kOutputTooSmall;
    std::lock_guard<std::mutex> lock(mutex_);
    *bound = model_bound_;
    std::copy(model_id_.begin(), model_id_.end(), model_id);
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::ReadState(float* out, std::size_t count) const {
    if (out == nullptr) return RuntimeStatus::kNullArgument;
    std::lock_guard<std::mutex> lock(mutex_);
    if (count != state_.size()) return RuntimeStatus::kCheckpointMismatch;
    std::copy(state_.begin(), state_.end(), out);
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::WriteState(const float* state, std::size_t count) {
    if (state == nullptr) return RuntimeStatus::kNullArgument;
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ == RuntimeLifecycle::kActive) return RuntimeStatus::kInvalidLifecycle;
    if (count != state_.size()) return RuntimeStatus::kCheckpointMismatch;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(state[i])) return RuntimeStatus::kInvalidConfig;
    }
    std::copy(state, state + count, state_.begin());
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::CheckpointSize(std::size_t* out) const {
    if (out == nullptr) return RuntimeStatus::kNullArgument;
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kSuspended) return RuntimeStatus::kInvalidLifecycle;
    const std::size_t header_size = model_bound_ ? kHeaderSizeV2 : kHeaderSizeV1;
    if (state_.size() > (std::numeric_limits<std::size_t>::max() - header_size) / 4) {
        return RuntimeStatus::kSizeOverflow;
    }
    *out = header_size + state_.size() * 4;
    return RuntimeStatus::kOk;
}
RuntimeStatus RuntimeSession::WriteCheckpoint(
    std::uint8_t* out,
    std::size_t capacity,
    std::size_t* written) const {
    if (out == nullptr || written == nullptr) return RuntimeStatus::kNullArgument;
    *written = 0;
    std::lock_guard<std::mutex> lock(mutex_);
    if (lifecycle_ != RuntimeLifecycle::kSuspended) return RuntimeStatus::kInvalidLifecycle;

    const std::size_t header_size = model_bound_ ? kHeaderSizeV2 : kHeaderSizeV1;
    const std::size_t required = header_size + state_.size() * 4;
    if (capacity < required) return RuntimeStatus::kOutputTooSmall;

    std::fill(out, out + required, 0);
    const auto& magic = model_bound_ ? kMagicV2 : kMagicV1;
    std::copy(magic.begin(), magic.end(), out);
    WriteU16(out + 8, model_bound_ ? kVersionV2 : kVersionV1);
    WriteU16(
        out + 10,
        static_cast<std::uint16_t>(header_size));
    WriteU32(out + 12, config_.layers);
    WriteU32(out + 16, config_.batch);
    WriteU32(out + 20, config_.d_model);
    WriteU32(out + 24, config_.d_state);
    WriteU32(out + 28, static_cast<std::uint32_t>(config_.recurrent_backend));
    WriteU32(out + 32, static_cast<std::uint32_t>(config_.packed_backend));
    WriteU32(out + 36, static_cast<std::uint32_t>(RuntimeLifecycle::kSuspended));
    WriteU64(out + 40, sequence_position_);
    WriteU64(out + 48, static_cast<std::uint64_t>(state_.size()));

    for (std::size_t i = 0; i < state_.size(); ++i) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(float));
        std::memcpy(&bits, &state_[i], sizeof(bits));
        WriteU32(out + header_size + i * 4, bits);
    }

    const std::uint32_t payload_crc =
        Crc32(out + header_size, state_.size() * 4);
    WriteU32(out + 56, payload_crc);

    if (model_bound_) {
        WriteU32(out + 60, kCheckpointModelBound);
        std::copy(model_id_.begin(), model_id_.end(), out + 64);
        WriteU32(out + 96, Crc32(out, 96));
    } else {
        WriteU32(out + 60, Crc32(out, 60));
    }

    *written = required;
    return RuntimeStatus::kOk;
}
RuntimeInfo RuntimeSession::Info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RuntimeInfo info;
    info.config = config_;
    info.resolved_recurrent_backend = resolved_recurrent_backend_;
    info.resolved_packed_backend = resolved_packed_backend_;
    info.lifecycle = lifecycle_;
    info.state_count = state_.size();
    info.sequence_position = sequence_position_;
    return info;
}

RuntimeStatus RuntimeSession::Restore(
    const std::uint8_t* blob,
    std::size_t blob_size,
    RuntimeSession** out) {
    if (blob == nullptr || out == nullptr) return RuntimeStatus::kNullArgument;
    *out = nullptr;
    if (blob_size < kHeaderSizeV1) return RuntimeStatus::kCheckpointCorrupt;

    const bool is_v1 =
        std::equal(kMagicV1.begin(), kMagicV1.end(), blob) &&
        ReadU16(blob + 8) == kVersionV1 &&
        ReadU16(blob + 10) == kHeaderSizeV1;
    const bool is_v2 =
        blob_size >= kHeaderSizeV2 &&
        std::equal(kMagicV2.begin(), kMagicV2.end(), blob) &&
        ReadU16(blob + 8) == kVersionV2 &&
        ReadU16(blob + 10) == kHeaderSizeV2;
    if (!is_v1 && !is_v2) return RuntimeStatus::kCheckpointCorrupt;

    const std::size_t header_size =
        is_v2 ? kHeaderSizeV2 : kHeaderSizeV1;
    if (is_v1) {
        if (ReadU32(blob + 60) != Crc32(blob, 60)) {
            return RuntimeStatus::kCheckpointCorrupt;
        }
    } else {
        if (ReadU32(blob + 96) != Crc32(blob, 96)) {
            return RuntimeStatus::kCheckpointCorrupt;
        }
        const std::uint32_t flags = ReadU32(blob + 60);
        if ((flags & ~kCheckpointModelBound) != 0) {
            return RuntimeStatus::kCheckpointCorrupt;
        }
        const bool bound = (flags & kCheckpointModelBound) != 0;
        const bool any_id = std::any_of(
            blob + 64,
            blob + 96,
            [](std::uint8_t value) { return value != 0; });
        if (bound != any_id) {
            return RuntimeStatus::kCheckpointCorrupt;
        }
    }

    RuntimeConfig config;
    config.layers = ReadU32(blob + 12);
    config.batch = ReadU32(blob + 16);
    config.d_model = ReadU32(blob + 20);
    config.d_state = ReadU32(blob + 24);
    config.recurrent_backend =
        static_cast<RecurrentBackend>(ReadU32(blob + 28));
    config.packed_backend =
        static_cast<PackedTernaryBackend>(ReadU32(blob + 32));
    if (!ValidRecurrentBackend(config.recurrent_backend) ||
        !ValidPackedBackend(config.packed_backend)) {
        return RuntimeStatus::kCheckpointCorrupt;
    }
    if (ReadU32(blob + 36) !=
        static_cast<std::uint32_t>(RuntimeLifecycle::kSuspended)) {
        return RuntimeStatus::kCheckpointCorrupt;
    }

    std::size_t count = 0;
    RuntimeStatus status = StateCount(config, &count);
    if (status != RuntimeStatus::kOk) {
        return RuntimeStatus::kCheckpointCorrupt;
    }
    if (ReadU64(blob + 48) != count) {
        return RuntimeStatus::kCheckpointMismatch;
    }
    if (count >
            (std::numeric_limits<std::size_t>::max() - header_size) / 4 ||
        blob_size != header_size + count * 4) {
        return RuntimeStatus::kCheckpointMismatch;
    }
    if (ReadU32(blob + 56) !=
        Crc32(blob + header_size, count * 4)) {
        return RuntimeStatus::kCheckpointCorrupt;
    }

    RuntimeSession* session = nullptr;
    status = Create(config, &session);
    if (status != RuntimeStatus::kOk) return status;

    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits =
            ReadU32(blob + header_size + i * 4);
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(bits));
        if (!std::isfinite(value)) {
            delete session;
            return RuntimeStatus::kCheckpointCorrupt;
        }
        session->state_[i] = value;
    }

    session->sequence_position_ = ReadU64(blob + 40);
    session->lifecycle_ = RuntimeLifecycle::kSuspended;
    if (is_v2 &&
        (ReadU32(blob + 60) & kCheckpointModelBound) != 0) {
        session->model_bound_ = true;
        std::copy(blob + 64, blob + 96, session->model_id_.begin());
    }
    *out = session;
    return RuntimeStatus::kOk;
}

}  // namespace vn97

namespace {
std::mutex g_registry_mutex;
std::unordered_map<std::uint64_t, std::shared_ptr<vn97::RuntimeSession>> g_sessions;
std::uint64_t g_next_handle = 1;

std::shared_ptr<vn97::RuntimeSession> Lookup(std::uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_sessions.find(handle);
    return it == g_sessions.end() ? nullptr : it->second;
}

int AddSession(vn97::RuntimeSession* raw, std::uint64_t* handle_out) {
    if (raw == nullptr || handle_out == nullptr) return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    std::shared_ptr<vn97::RuntimeSession> session(raw);
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    if (g_next_handle == 0) return static_cast<int>(vn97::RuntimeStatus::kCounterOverflow);
    const std::uint64_t handle = g_next_handle++;
    g_sessions.emplace(handle, std::move(session));
    *handle_out = handle;
    return static_cast<int>(vn97::RuntimeStatus::kOk);
}

bool DecodeRecurrent(int value, vn97::RecurrentBackend* out) {
    if (out == nullptr) return false;
    if (value < 0 || value > 2) return false;
    *out = static_cast<vn97::RecurrentBackend>(value);
    return true;
}
bool DecodePacked(int value, vn97::PackedTernaryBackend* out) {
    if (out == nullptr) return false;
    if (value < 0 || value > 2) return false;
    *out = static_cast<vn97::PackedTernaryBackend>(value);
    return true;
}
}

extern "C" {
int vn97_runtime_create(const vn97_runtime_config* config, std::uint64_t* handle_out) {
    if (config == nullptr || handle_out == nullptr) return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    vn97::RuntimeConfig cpp;
    cpp.layers=config->layers; cpp.batch=config->batch; cpp.d_model=config->d_model; cpp.d_state=config->d_state;
    if (!DecodeRecurrent(config->recurrent_backend,&cpp.recurrent_backend) || !DecodePacked(config->packed_backend,&cpp.packed_backend)) return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    vn97::RuntimeSession* raw=nullptr;
    const auto status=vn97::RuntimeSession::Create(cpp,&raw);
    if(status!=vn97::RuntimeStatus::kOk) return static_cast<int>(status);
    return AddSession(raw,handle_out);
}
int vn97_runtime_restore(const std::uint8_t* blob,std::size_t blob_size,std::uint64_t* handle_out){ vn97::RuntimeSession* raw=nullptr; const auto s=vn97::RuntimeSession::Restore(blob,blob_size,&raw); if(s!=vn97::RuntimeStatus::kOk) return static_cast<int>(s); return AddSession(raw,handle_out); }
int vn97_runtime_destroy(std::uint64_t handle){ std::lock_guard<std::mutex> lock(g_registry_mutex); return g_sessions.erase(handle)==1?0:static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_activate(std::uint64_t h){ auto s=Lookup(h); return s?static_cast<int>(s->Activate()):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_suspend(std::uint64_t h){ auto s=Lookup(h); return s?static_cast<int>(s->Suspend()):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_resume(std::uint64_t h){ auto s=Lookup(h); return s?static_cast<int>(s->Resume()):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_advance(std::uint64_t h,std::uint64_t n){ auto s=Lookup(h); return s?static_cast<int>(s->Advance(n)):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_model_binding_get(std::uint64_t h,int* bound,std::uint8_t* model_id,std::size_t capacity){ if(!bound||!model_id) return static_cast<int>(vn97::RuntimeStatus::kNullArgument); auto s=Lookup(h); if(!s) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); bool cpp_bound=false; const auto status=s->ModelBinding(&cpp_bound,model_id,capacity); if(status==vn97::RuntimeStatus::kOk) *bound=cpp_bound?1:0; return static_cast<int>(status); }
int vn97_runtime_info_get(std::uint64_t h,vn97_runtime_info* out){ if(!out) return static_cast<int>(vn97::RuntimeStatus::kNullArgument); auto s=Lookup(h); if(!s) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); const auto i=s->Info(); out->layers=i.config.layers; out->batch=i.config.batch; out->d_model=i.config.d_model; out->d_state=i.config.d_state; out->requested_recurrent_backend=static_cast<int>(i.config.recurrent_backend); out->requested_packed_backend=static_cast<int>(i.config.packed_backend); out->resolved_recurrent_backend=static_cast<int>(i.resolved_recurrent_backend); out->resolved_packed_backend=static_cast<int>(i.resolved_packed_backend); out->lifecycle=static_cast<int>(i.lifecycle); out->state_count=i.state_count; out->sequence_position=i.sequence_position; return 0; }
int vn97_runtime_state_read(std::uint64_t h,float* out,std::size_t n){ auto s=Lookup(h); return s?static_cast<int>(s->ReadState(out,n)):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_state_write(std::uint64_t h,const float* in,std::size_t n){ auto s=Lookup(h); return s?static_cast<int>(s->WriteState(in,n)):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_checkpoint_size(std::uint64_t h,std::size_t* out){ auto s=Lookup(h); return s?static_cast<int>(s->CheckpointSize(out)):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_checkpoint_write(std::uint64_t h,std::uint8_t* out,std::size_t c,std::size_t* w){ auto s=Lookup(h); return s?static_cast<int>(s->WriteCheckpoint(out,c,w)):static_cast<int>(vn97::RuntimeStatus::kInvalidHandle); }
int vn97_runtime_infer_step(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    float* logits,
    std::size_t logits_count) {
    if (language_model_view == nullptr || input_ids == nullptr || logits == nullptr) {
        return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    }
    const auto session = Lookup(handle);
    if (!session) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle);
    const auto info = session->Info();
    if (input_count != info.config.batch) {
        return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    }
    return static_cast<int>(session->InferStep(*language_model_view, input_ids, logits, logits_count));
}

int vn97_runtime_infer_step_hidden(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    float* hidden,
    std::size_t hidden_count) {
    if (language_model_view == nullptr || input_ids == nullptr || hidden == nullptr) {
        return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    }
    const auto session = Lookup(handle);
    if (!session) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle);
    const auto info = session->Info();
    if (input_count != info.config.batch) {
        return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    }
    return static_cast<int>(
        session->InferStepHidden(*language_model_view, input_ids, hidden, hidden_count));
}

int vn97_runtime_prefill(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    std::size_t step_count,
    float* final_logits,
    std::size_t logits_count) {
    if (language_model_view == nullptr || input_ids == nullptr || final_logits == nullptr) {
        return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    }
    if (step_count == 0) return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    const auto session = Lookup(handle);
    if (!session) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle);
    const auto info = session->Info();
    const std::size_t batch = info.config.batch;
    if (batch == 0 || step_count > std::numeric_limits<std::size_t>::max() / batch) {
        return static_cast<int>(vn97::RuntimeStatus::kSizeOverflow);
    }
    if (input_count != step_count * batch) {
        return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    }
    for (std::size_t step = 0; step < step_count; ++step) {
        const auto status = session->InferStep(
            *language_model_view,
            input_ids + step * batch,
            final_logits,
            logits_count);
        if (status != vn97::RuntimeStatus::kOk) return static_cast<int>(status);
    }
    return static_cast<int>(vn97::RuntimeStatus::kOk);
}

int vn97_runtime_prefill_hidden(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    std::size_t step_count,
    float* final_hidden,
    std::size_t hidden_count) {
    if (language_model_view == nullptr || input_ids == nullptr || final_hidden == nullptr) {
        return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    }
    if (step_count == 0) return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    const auto session = Lookup(handle);
    if (!session) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle);
    const auto info = session->Info();
    const std::size_t batch = info.config.batch;
    if (batch == 0 || step_count > std::numeric_limits<std::size_t>::max() / batch) {
        return static_cast<int>(vn97::RuntimeStatus::kSizeOverflow);
    }
    if (input_count != step_count * batch) {
        return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    }
    for (std::size_t step = 0; step < step_count; ++step) {
        const auto status = session->InferStepHidden(
            *language_model_view,
            input_ids + step * batch,
            final_hidden,
            hidden_count);
        if (status != vn97::RuntimeStatus::kOk) return static_cast<int>(status);
    }
    return static_cast<int>(vn97::RuntimeStatus::kOk);
}

int vn97_runtime_generate_greedy(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* prompt_ids,
    std::size_t prompt_count,
    std::size_t max_new_tokens,
    std::uint32_t eos_token,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count) {
    if (output_count == nullptr || language_model_view == nullptr) {
        return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    }
    *output_count = 0;
    if (max_new_tokens == 0) return static_cast<int>(vn97::RuntimeStatus::kOk);
    if (prompt_ids == nullptr || output_ids == nullptr) {
        return static_cast<int>(vn97::RuntimeStatus::kNullArgument);
    }
    if (prompt_count == 0) return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    if (output_capacity < max_new_tokens) {
        return static_cast<int>(vn97::RuntimeStatus::kOutputTooSmall);
    }

    const auto session = Lookup(handle);
    if (!session) return static_cast<int>(vn97::RuntimeStatus::kInvalidHandle);
    const auto info = session->Info();
    if (info.config.batch != 1) {
        return static_cast<int>(vn97::RuntimeStatus::kInvalidConfig);
    }

    if (language_model_view->vocab_size <= 1) {
        return static_cast<int>(vn97::RuntimeStatus::kInferenceError);
    }
    std::vector<float> logits;
    try {
        logits.assign(language_model_view->vocab_size, 0.0f);
    } catch (...) {
        return static_cast<int>(vn97::RuntimeStatus::kSizeOverflow);
    }

    for (std::size_t i = 0; i < prompt_count; ++i) {
        const auto status = session->InferStep(
            *language_model_view,
            prompt_ids + i,
            logits.data(),
            logits.size());
        if (status != vn97::RuntimeStatus::kOk) return static_cast<int>(status);
    }

    for (std::size_t generated = 0; generated < max_new_tokens; ++generated) {
        std::uint32_t token = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (std::uint32_t id = 0; id < language_model_view->vocab_size; ++id) {
            const float value = logits[id];
            if (!std::isfinite(value)) {
                return static_cast<int>(vn97::RuntimeStatus::kInferenceError);
            }
            if (id == 0 || value > best) {
                best = value;
                token = id;
            }
        }

        output_ids[generated] = token;
        const auto status = session->InferStep(
            *language_model_view,
            &token,
            logits.data(),
            logits.size());
        if (status != vn97::RuntimeStatus::kOk) return static_cast<int>(status);
        *output_count = generated + 1;
        if (token == eos_token) break;
    }
    return static_cast<int>(vn97::RuntimeStatus::kOk);
}

}
