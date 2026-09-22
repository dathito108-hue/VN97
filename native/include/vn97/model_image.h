#pragma once

#include "vn97/language.h"
#include "vn97/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace vn97 {

constexpr std::size_t kActivatedModelIdentityBytes = 32;
constexpr std::size_t kActivatedModelMaxBytes = 512ull * 1024ull * 1024ull;

enum class ModelImageStatus {
    kOk = 0,
    kNullArgument,
    kInvalidHandle,
    kInvalidFd,
    kInvalidRange,
    kTooLarge,
    kMapFailed,
    kIdentityMismatch,
    kBadMagic,
    kUnsupportedVersion,
    kInvalidHeader,
    kInvalidSection,
    kInvalidConfig,
    kInvalidPackedTernary,
    kInvalidTokenizer,
    kInvalidModel,
    kOutOfMemory,
    kIoError,
    kOutputTooSmall,
};

struct ActivatedModelInfo {
    std::uint8_t model_id[kActivatedModelIdentityBytes] = {};
    std::uint32_t vocab_size = 0;
    std::uint32_t d_model = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t d_state = 0;
    LanguageEmbeddingKind embedding_kind = LanguageEmbeddingKind::kFullF32;
    std::uint32_t embedding_rank = 0;
    bool has_tokenizer = false;
    std::size_t image_bytes = 0;
};

class ActivatedModelImage {
public:
    ~ActivatedModelImage();
    ActivatedModelImage(const ActivatedModelImage&) = delete;
    ActivatedModelImage& operator=(const ActivatedModelImage&) = delete;

    static ModelImageStatus OpenFd(
        int fd,
        std::uint64_t offset,
        std::uint64_t length,
        const std::uint8_t expected_model_id[kActivatedModelIdentityBytes],
        std::unique_ptr<ActivatedModelImage>* out);

    const LanguageModelView& language_model() const { return language_model_; }
    const TokenizerView* tokenizer() const {
        return has_tokenizer_ ? &tokenizer_ : nullptr;
    }
    ActivatedModelInfo Info() const;

private:
    ActivatedModelImage() = default;

    ModelImageStatus Parse(
        const std::uint8_t expected_model_id[kActivatedModelIdentityBytes]);

    int fd_ = -1;
    void* mapping_base_ = nullptr;
    std::size_t mapping_bytes_ = 0;
    const std::uint8_t* image_ = nullptr;
    std::size_t image_bytes_ = 0;
    LanguageModelView language_model_;
    TokenizerView tokenizer_;
    bool has_tokenizer_ = false;
    std::unique_ptr<LanguageLayerView[]> layers_;
    std::unique_ptr<float[]> stable_a_;
};

}  // namespace vn97

extern "C" {

struct vn97_model_info {
    std::uint8_t model_id[32];
    std::uint32_t vocab_size;
    std::uint32_t d_model;
    std::uint32_t n_layers;
    std::uint32_t d_state;
    std::uint32_t embedding_kind;
    std::uint32_t embedding_rank;
    int has_tokenizer;
    std::size_t image_bytes;
};

int vn97_model_open_fd(
    int fd,
    std::uint64_t offset,
    std::uint64_t length,
    const std::uint8_t* expected_model_id,
    std::size_t expected_model_id_size,
    std::uint64_t* handle_out);

int vn97_model_destroy(std::uint64_t handle);
int vn97_model_info_get(std::uint64_t handle, vn97_model_info* out);
// These inference entry points keep the model registry reference alive for the
// entire runtime call. Their return values use vn97::RuntimeStatus codes.
int vn97_model_runtime_infer_step(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    float* logits,
    std::size_t logits_count);

int vn97_model_runtime_prefill(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    std::size_t step_count,
    float* final_logits,
    std::size_t logits_count);

int vn97_model_runtime_generate_greedy(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* prompt_ids,
    std::size_t prompt_count,
    std::size_t max_new_tokens,
    std::uint32_t eos_token,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count);

int vn97_model_tokenizer_encode(
    std::uint64_t handle,
    const std::uint8_t* input,
    std::size_t input_size,
    std::uint32_t flags,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count);

int vn97_model_tokenizer_decoded_size(
    std::uint64_t handle,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    int skip_control,
    std::size_t* output_size);

int vn97_model_tokenizer_decode(
    std::uint64_t handle,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    int skip_control,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* output_size);

}
