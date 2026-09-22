#pragma once

#include "vn97/language.h"
#include "vn97/packed_ternary.h"
#include "vn97/recurrent.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace vn97 {

enum class RuntimeStatus {
    kOk = 0,
    kNullArgument,
    kInvalidConfig,
    kSizeOverflow,
    kInvalidHandle,
    kInvalidLifecycle,
    kOutputTooSmall,
    kCheckpointCorrupt,
    kCheckpointMismatch,
    kCounterOverflow,
    kBackendUnavailable,
    kModelMismatch,
    kInferenceError,
};

enum class RuntimeLifecycle : std::uint32_t {
    kCreated = 0,
    kActive = 1,
    kSuspended = 2,
};

struct RuntimeConfig {
    std::uint32_t layers = 0;
    std::uint32_t batch = 0;
    std::uint32_t d_model = 0;
    std::uint32_t d_state = 0;
    RecurrentBackend recurrent_backend = RecurrentBackend::kAuto;
    PackedTernaryBackend packed_backend = PackedTernaryBackend::kAuto;
};

struct RuntimeInfo {
    RuntimeConfig config;
    RecurrentBackend resolved_recurrent_backend = RecurrentBackend::kScalar;
    PackedTernaryBackend resolved_packed_backend = PackedTernaryBackend::kScalar;
    RuntimeLifecycle lifecycle = RuntimeLifecycle::kCreated;
    std::size_t state_count = 0;
    std::uint64_t sequence_position = 0;
};

class RuntimeSession {
public:
    ~RuntimeSession() = default;
    RuntimeSession(const RuntimeSession&) = delete;
    RuntimeSession& operator=(const RuntimeSession&) = delete;

    static RuntimeStatus Create(
        const RuntimeConfig& config,
        RuntimeSession** out);

    static RuntimeStatus Restore(
        const std::uint8_t* blob,
        std::size_t blob_size,
        RuntimeSession** out);

    RuntimeStatus Activate();
    RuntimeStatus Suspend();
    RuntimeStatus Resume();
    RuntimeStatus Advance(std::uint64_t token_count);
    RuntimeStatus InferStep(
        const LanguageModelView& model,
        const std::uint32_t* input_ids,
        float* logits,
        std::size_t logits_count);
    RuntimeStatus ModelBinding(
        bool* bound,
        std::uint8_t* model_id,
        std::size_t model_id_capacity) const;
    RuntimeStatus ReadState(float* out, std::size_t count) const;
    RuntimeStatus WriteState(const float* state, std::size_t count);
    RuntimeStatus CheckpointSize(std::size_t* out) const;
    RuntimeStatus WriteCheckpoint(
        std::uint8_t* out,
        std::size_t capacity,
        std::size_t* written) const;
    RuntimeInfo Info() const;

private:
    RuntimeSession() = default;

    RuntimeConfig config_;
    RecurrentBackend resolved_recurrent_backend_ = RecurrentBackend::kScalar;
    PackedTernaryBackend resolved_packed_backend_ = PackedTernaryBackend::kScalar;
    RuntimeLifecycle lifecycle_ = RuntimeLifecycle::kCreated;
    std::uint64_t sequence_position_ = 0;
    bool model_bound_ = false;
    std::array<std::uint8_t, 32> model_id_ = {};
    std::vector<float> state_;
    std::vector<float> language_workspace_;
    mutable std::mutex mutex_;
};

}  // namespace vn97

extern "C" {

struct vn97_runtime_config {
    std::uint32_t layers;
    std::uint32_t batch;
    std::uint32_t d_model;
    std::uint32_t d_state;
    int recurrent_backend;
    int packed_backend;
};

struct vn97_runtime_info {
    std::uint32_t layers;
    std::uint32_t batch;
    std::uint32_t d_model;
    std::uint32_t d_state;
    int requested_recurrent_backend;
    int requested_packed_backend;
    int resolved_recurrent_backend;
    int resolved_packed_backend;
    int lifecycle;
    std::size_t state_count;
    std::uint64_t sequence_position;
};

int vn97_runtime_create(
    const vn97_runtime_config* config,
    std::uint64_t* handle_out);

int vn97_runtime_restore(
    const std::uint8_t* blob,
    std::size_t blob_size,
    std::uint64_t* handle_out);

int vn97_runtime_destroy(std::uint64_t handle);
int vn97_runtime_activate(std::uint64_t handle);
int vn97_runtime_suspend(std::uint64_t handle);
int vn97_runtime_resume(std::uint64_t handle);
int vn97_runtime_advance(std::uint64_t handle, std::uint64_t token_count);
int vn97_runtime_model_binding_get(
    std::uint64_t handle,
    int* bound,
    std::uint8_t* model_id,
    std::size_t model_id_capacity);
int vn97_runtime_info_get(std::uint64_t handle, vn97_runtime_info* out);
int vn97_runtime_state_read(
    std::uint64_t handle,
    float* out,
    std::size_t count);
int vn97_runtime_state_write(
    std::uint64_t handle,
    const float* state,
    std::size_t count);
int vn97_runtime_checkpoint_size(
    std::uint64_t handle,
    std::size_t* out);
int vn97_runtime_checkpoint_write(
    std::uint64_t handle,
    std::uint8_t* out,
    std::size_t capacity,
    std::size_t* written);

int vn97_runtime_infer_step(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    float* logits,
    std::size_t logits_count);

int vn97_runtime_prefill(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    std::size_t step_count,
    float* final_logits,
    std::size_t logits_count);

int vn97_runtime_generate_greedy(
    std::uint64_t handle,
    const vn97::LanguageModelView* language_model_view,
    const std::uint32_t* prompt_ids,
    std::size_t prompt_count,
    std::size_t max_new_tokens,
    std::uint32_t eos_token,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count);

}
