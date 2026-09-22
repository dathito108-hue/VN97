#pragma once

#include <cstddef>
#include <cstdint>

namespace vn97 {

constexpr std::uint32_t kTokenizerControlCount = 8;
constexpr std::uint32_t kTokenizerByteBase = 8;
constexpr std::uint32_t kTokenizerLearnedBase = 264;

enum class TokenizerStatus {
    kOk = 0,
    kNullArgument,
    kBlobTooShort,
    kBadMagic,
    kUnsupportedVersion,
    kInvalidLayout,
    kTruncatedTable,
    kInvalidToken,
    kDuplicateToken,
    kOutputTooSmall,
    kTokenOutOfRange,
    kControlToken,
};

enum TokenizerEncodeFlags : std::uint32_t {
    kTokenizerAddBos = 1u << 0,
    kTokenizerAddText = 1u << 1,
    kTokenizerAddEos = 1u << 2,
};

struct TokenizerView {
    const std::uint8_t* blob = nullptr;
    std::size_t blob_size = 0;
    const std::uint8_t* offset_table = nullptr;
    const std::uint8_t* token_data = nullptr;
    std::size_t token_data_size = 0;
    std::uint32_t learned_count = 0;
};

TokenizerStatus ParseTokenizer(
    const std::uint8_t* blob,
    std::size_t blob_size,
    TokenizerView* out);

TokenizerStatus TokenizerEncodeBytes(
    const TokenizerView& tokenizer,
    const std::uint8_t* input,
    std::size_t input_size,
    std::uint32_t flags,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count);

TokenizerStatus TokenizerDecodeBytes(
    const TokenizerView& tokenizer,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    bool skip_control,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* output_size);

}  // namespace vn97

extern "C" {

int vn97_tokenizer_encode_bytes(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const std::uint8_t* input,
    std::size_t input_size,
    std::uint32_t flags,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count);

int vn97_tokenizer_decode_bytes(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    int skip_control,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* output_size);

}
