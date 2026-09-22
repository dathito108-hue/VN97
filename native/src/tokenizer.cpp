#include "vn97/tokenizer.h"

#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace vn97 {
namespace {

constexpr std::uint8_t kMagic[8] = {'V', 'N', '9', '7', 'T', 'K', '1', 0};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 24;
constexpr std::size_t kBucketStartCount = kTokenizerBucketCount + 1;

std::uint32_t ReadU32LE(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
}

bool AddOverflows(std::size_t a, std::size_t b) {
    return a > std::numeric_limits<std::size_t>::max() - b;
}

bool TokenAt(
    const TokenizerView& tokenizer,
    std::uint32_t learned_index,
    const std::uint8_t** bytes,
    std::uint32_t* size) {
    if (learned_index >= tokenizer.learned_count) return false;
    const std::size_t offset_pos =
        static_cast<std::size_t>(learned_index) * 4;
    const std::uint32_t start =
        ReadU32LE(tokenizer.offset_table + offset_pos);
    const std::uint32_t end =
        ReadU32LE(tokenizer.offset_table + offset_pos + 4);
    if (end < start || end > tokenizer.token_data_size) return false;
    *bytes = tokenizer.token_data + start;
    *size = end - start;
    return true;
}

}  // namespace

TokenizerStatus ParseTokenizer(
    const std::uint8_t* blob,
    std::size_t blob_size,
    TokenizerView* out) {
    if (blob == nullptr || out == nullptr) {
        return TokenizerStatus::kNullArgument;
    }
    if (blob_size < kHeaderSize) {
        return TokenizerStatus::kBlobTooShort;
    }
    if (std::memcmp(blob, kMagic, sizeof(kMagic)) != 0) {
        return TokenizerStatus::kBadMagic;
    }
    if (ReadU32LE(blob + 8) != kVersion) {
        return TokenizerStatus::kUnsupportedVersion;
    }
    if (ReadU32LE(blob + 12) != kTokenizerControlCount ||
        ReadU32LE(blob + 16) != kTokenizerByteBase) {
        return TokenizerStatus::kInvalidLayout;
    }

    TokenizerView view;
    view.blob = blob;
    view.blob_size = blob_size;
    view.learned_count = ReadU32LE(blob + 20);

    const std::size_t offset_count =
        static_cast<std::size_t>(view.learned_count) + 1;
    const std::size_t bucket_index_count =
        static_cast<std::size_t>(view.learned_count);
    if (
        MulOverflows(offset_count, 4) ||
        MulOverflows(kBucketStartCount, 4) ||
        MulOverflows(bucket_index_count, 4)
    ) {
        return TokenizerStatus::kInvalidLayout;
    }

    const std::size_t offset_bytes =
        offset_count * 4;
    const std::size_t bucket_start_bytes =
        kBucketStartCount * 4;
    const std::size_t bucket_index_bytes =
        bucket_index_count * 4;

    if (
        AddOverflows(kHeaderSize, offset_bytes) ||
        AddOverflows(
            kHeaderSize + offset_bytes,
            bucket_start_bytes) ||
        AddOverflows(
            kHeaderSize +
                offset_bytes +
                bucket_start_bytes,
            bucket_index_bytes)
    ) {
        return TokenizerStatus::kInvalidLayout;
    }

    const std::size_t table_end =
        kHeaderSize +
        offset_bytes +
        bucket_start_bytes +
        bucket_index_bytes;
    if (table_end > blob_size) {
        return TokenizerStatus::kTruncatedTable;
    }

    view.offset_table =
        blob + kHeaderSize;
    view.bucket_starts =
        view.offset_table + offset_bytes;
    view.bucket_indices =
        view.bucket_starts + bucket_start_bytes;
    view.token_data =
        view.bucket_indices + bucket_index_bytes;
    view.token_data_size =
        blob_size - table_end;

    if (ReadU32LE(view.offset_table) != 0) {
        return TokenizerStatus::kInvalidLayout;
    }
    const std::uint32_t final_offset =
        ReadU32LE(
            view.offset_table +
            static_cast<std::size_t>(
                view.learned_count) * 4);
    if (final_offset != view.token_data_size) {
        return TokenizerStatus::kInvalidLayout;
    }

    std::unordered_set<
        std::string_view
    > seen_tokens;
    seen_tokens.reserve(
        view.learned_count
    );

    for (
        std::uint32_t index = 0;
        index < view.learned_count;
        ++index
    ) {
        const std::uint8_t* token = nullptr;
        std::uint32_t size = 0;
        if (
            !TokenAt(
                view,
                index,
                &token,
                &size)
            || size < 2
        ) {
            return TokenizerStatus::kInvalidToken;
        }

        const std::string_view key(
            reinterpret_cast<
                const char*
            >(token),
            size);

        if (!seen_tokens.insert(key).second) {
            return TokenizerStatus::kDuplicateToken;
        }
    }

    if (
        ReadU32LE(
            view.bucket_starts)
        != 0
    ) {
        return TokenizerStatus::kInvalidLayout;
    }
    const std::uint32_t final_bucket =
        ReadU32LE(
            view.bucket_starts +
            kTokenizerBucketCount * 4);
    if (final_bucket != view.learned_count) {
        return TokenizerStatus::kInvalidLayout;
    }

    std::vector<std::uint8_t> seen_indices(
        view.learned_count,
        0);

    for (
        std::uint32_t first = 0;
        first < kTokenizerBucketCount;
        ++first
    ) {
        const std::uint32_t start =
            ReadU32LE(
                view.bucket_starts +
                first * 4);
        const std::uint32_t end =
            ReadU32LE(
                view.bucket_starts +
                (first + 1) * 4);

        if (
            end < start ||
            end > view.learned_count
        ) {
            return TokenizerStatus::kInvalidLayout;
        }

        std::uint32_t previous_size =
            std::numeric_limits<
                std::uint32_t
            >::max();
        std::uint32_t previous_index = 0;
        bool has_previous = false;

        for (
            std::uint32_t position = start;
            position < end;
            ++position
        ) {
            const std::uint32_t index =
                ReadU32LE(
                    view.bucket_indices +
                    static_cast<std::size_t>(
                        position) * 4);

            if (
                index >= view.learned_count ||
                seen_indices[index]
            ) {
                return TokenizerStatus::kInvalidLayout;
            }
            seen_indices[index] = 1;

            const std::uint8_t* token = nullptr;
            std::uint32_t size = 0;

            if (
                !TokenAt(
                    view,
                    index,
                    &token,
                    &size)
                || token[0] != first
            ) {
                return TokenizerStatus::kInvalidLayout;
            }

            if (
                has_previous &&
                (
                    size > previous_size ||
                    (
                        size == previous_size &&
                        index < previous_index
                    )
                )
            ) {
                return TokenizerStatus::kInvalidLayout;
            }

            previous_size = size;
            previous_index = index;
            has_previous = true;
        }
    }

    *out = view;
    return TokenizerStatus::kOk;
}

TokenizerStatus TokenizerEncodeBytes(
    const TokenizerView& tokenizer,
    const std::uint8_t* input,
    std::size_t input_size,
    std::uint32_t flags,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count) {
    if (
        output_count == nullptr ||
        (
            input_size != 0 &&
            input == nullptr
        ) ||
        (
            output_capacity != 0 &&
            output_ids == nullptr
        )
    ) {
        return TokenizerStatus::kNullArgument;
    }

    if (
        flags &
        ~(
            kTokenizerAddBos |
            kTokenizerAddText |
            kTokenizerAddEos
        )
    ) {
        return TokenizerStatus::kInvalidLayout;
    }

    std::size_t count = 0;
    auto emit = [&](
        std::uint32_t token_id
    ) -> bool {
        if (
            count >= output_capacity
        ) {
            return false;
        }
        output_ids[count++] = token_id;
        return true;
    };

    if (
        (flags & kTokenizerAddBos) &&
        !emit(1)
    ) {
        *output_count = count + 1;
        return TokenizerStatus::kOutputTooSmall;
    }

    if (
        (flags & kTokenizerAddText) &&
        !emit(3)
    ) {
        *output_count = count + 1;
        return TokenizerStatus::kOutputTooSmall;
    }

    std::size_t offset = 0;

    while (offset < input_size) {
        const std::uint32_t first =
            input[offset];

        const std::uint32_t start =
            ReadU32LE(
                tokenizer.bucket_starts +
                first * 4);
        const std::uint32_t end =
            ReadU32LE(
                tokenizer.bucket_starts +
                (first + 1) * 4);

        std::uint32_t matched_id =
            kTokenizerByteBase + first;
        std::uint32_t matched_size = 1;

        for (
            std::uint32_t position = start;
            position < end;
            ++position
        ) {
            const std::uint32_t index =
                ReadU32LE(
                    tokenizer.bucket_indices +
                    static_cast<std::size_t>(
                        position) * 4);

            const std::uint8_t* token = nullptr;
            std::uint32_t size = 0;

            if (!TokenAt(
                    tokenizer,
                    index,
                    &token,
                    &size)) {
                return TokenizerStatus::kInvalidToken;
            }

            if (
                offset + size <= input_size &&
                std::memcmp(
                    input + offset,
                    token,
                    size) == 0
            ) {
                matched_id =
                    kTokenizerLearnedBase + index;
                matched_size = size;
                break;
            }
        }

        if (!emit(matched_id)) {
            *output_count =
                count +
                (input_size - offset) +
                (
                    (
                        flags &
                        kTokenizerAddEos
                    )
                    ? 1u
                    : 0u
                );
            return TokenizerStatus::kOutputTooSmall;
        }

        offset += matched_size;
    }

    if (
        (flags & kTokenizerAddEos) &&
        !emit(2)
    ) {
        *output_count = count + 1;
        return TokenizerStatus::kOutputTooSmall;
    }

    *output_count = count;
    return TokenizerStatus::kOk;
}

TokenizerStatus TokenizerDecodeBytes(
    const TokenizerView& tokenizer,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    bool skip_control,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* output_size) {
    if (
        output_size == nullptr ||
        (
            token_count != 0 &&
            token_ids == nullptr
        ) ||
        (
            output_capacity != 0 &&
            output == nullptr
        )
    ) {
        return TokenizerStatus::kNullArgument;
    }

    std::size_t required = 0;

    for (
        std::size_t i = 0;
        i < token_count;
        ++i
    ) {
        const std::uint32_t token_id =
            token_ids[i];

        if (
            token_id <
            kTokenizerControlCount
        ) {
            if (!skip_control) {
                return TokenizerStatus::kControlToken;
            }
            continue;
        }

        std::size_t size = 0;

        if (
            token_id <
            kTokenizerLearnedBase
        ) {
            size = 1;
        } else {
            const std::uint32_t index =
                token_id -
                kTokenizerLearnedBase;
            const std::uint8_t* token =
                nullptr;
            std::uint32_t token_size = 0;

            if (!TokenAt(
                    tokenizer,
                    index,
                    &token,
                    &token_size)) {
                return TokenizerStatus::kTokenOutOfRange;
            }

            size = token_size;
        }

        if (
            AddOverflows(
                required,
                size)
        ) {
            return TokenizerStatus::kInvalidLayout;
        }

        required += size;
    }

    *output_size = required;

    if (
        required > output_capacity
    ) {
        return TokenizerStatus::kOutputTooSmall;
    }

    std::size_t offset = 0;

    for (
        std::size_t i = 0;
        i < token_count;
        ++i
    ) {
        const std::uint32_t token_id =
            token_ids[i];

        if (
            token_id <
            kTokenizerControlCount
        ) {
            continue;
        }

        if (
            token_id <
            kTokenizerLearnedBase
        ) {
            output[offset++] =
                static_cast<std::uint8_t>(
                    token_id -
                    kTokenizerByteBase);
            continue;
        }

        const std::uint32_t index =
            token_id -
            kTokenizerLearnedBase;
        const std::uint8_t* token =
            nullptr;
        std::uint32_t size = 0;

        if (!TokenAt(
                tokenizer,
                index,
                &token,
                &size)) {
            return TokenizerStatus::kTokenOutOfRange;
        }

        std::memcpy(
            output + offset,
            token,
            size);

        offset += size;
    }

    return TokenizerStatus::kOk;
}

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
    std::size_t* output_count) {
    vn97::TokenizerView tokenizer;

    const auto status =
        vn97::ParseTokenizer(
            blob,
            blob_size,
            &tokenizer);

    if (
        status !=
        vn97::TokenizerStatus::kOk
    ) {
        return static_cast<int>(
            status
        );
    }

    return static_cast<int>(
        vn97::TokenizerEncodeBytes(
            tokenizer,
            input,
            input_size,
            flags,
            output_ids,
            output_capacity,
            output_count));
}

int vn97_tokenizer_decode_bytes(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    int skip_control,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* output_size) {
    vn97::TokenizerView tokenizer;

    const auto status =
        vn97::ParseTokenizer(
            blob,
            blob_size,
            &tokenizer);

    if (
        status !=
        vn97::TokenizerStatus::kOk
    ) {
        return static_cast<int>(
            status
        );
    }

    return static_cast<int>(
        vn97::TokenizerDecodeBytes(
            tokenizer,
            token_ids,
            token_count,
            skip_control != 0,
            output,
            output_capacity,
            output_size));
}

}
