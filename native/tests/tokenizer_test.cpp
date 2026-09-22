#include "vn97/tokenizer.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void PushU32(
    std::vector<std::uint8_t>& out,
    std::uint32_t value) {
    out.push_back(
        static_cast<std::uint8_t>(
            value));
    out.push_back(
        static_cast<std::uint8_t>(
            value >> 8));
    out.push_back(
        static_cast<std::uint8_t>(
            value >> 16));
    out.push_back(
        static_cast<std::uint8_t>(
            value >> 24));
}

std::vector<std::uint8_t> BuildPackage(
    const std::vector<std::string>& tokens) {
    std::vector<std::uint8_t> blob = {
        'V', 'N', '9', '7',
        'T', 'K', '1', 0
    };

    PushU32(blob, 1);
    PushU32(blob, 8);
    PushU32(blob, 8);
    PushU32(
        blob,
        static_cast<std::uint32_t>(
            tokens.size()));

    std::uint32_t offset = 0;
    PushU32(blob, offset);

    for (
        const auto& token : tokens
    ) {
        offset +=
            static_cast<std::uint32_t>(
                token.size());
        PushU32(
            blob,
            offset);
    }

    std::array<
        std::vector<std::uint32_t>,
        256
    > buckets;

    for (
        std::uint32_t i = 0;
        i < tokens.size();
        ++i
    ) {
        buckets[
            static_cast<unsigned char>(
                tokens[i][0])
        ].push_back(i);
    }

    for (
        auto& bucket : buckets
    ) {
        std::sort(
            bucket.begin(),
            bucket.end(),
            [&](std::uint32_t a,
                std::uint32_t b) {
                if (
                    tokens[a].size()
                    != tokens[b].size()
                ) {
                    return (
                        tokens[a].size()
                        > tokens[b].size()
                    );
                }
                return a < b;
            });
    }

    std::uint32_t bucket_position = 0;
    PushU32(
        blob,
        bucket_position);

    for (
        const auto& bucket : buckets
    ) {
        bucket_position +=
            static_cast<std::uint32_t>(
                bucket.size());
        PushU32(
            blob,
            bucket_position);
    }

    for (
        const auto& bucket : buckets
    ) {
        for (
            std::uint32_t index : bucket
        ) {
            PushU32(
                blob,
                index);
        }
    }

    for (
        const auto& token : tokens
    ) {
        blob.insert(
            blob.end(),
            token.begin(),
            token.end());
    }

    return blob;
}

}  // namespace

int main() {
    auto blob = BuildPackage(
        {"ab", "abc", "xyz"});

    vn97::TokenizerView view;

    assert(
        vn97::ParseTokenizer(
            blob.data(),
            blob.size(),
            &view) ==
        vn97::TokenizerStatus::kOk);

    assert(
        view.learned_count == 3
    );

    const std::uint8_t input[] = {
        'a', 'b', 'c',
        'a', 'b', 'x'
    };

    std::uint32_t ids[9] = {};
    std::size_t count = 0;

    assert(
        vn97::TokenizerEncodeBytes(
            view,
            input,
            sizeof(input),
            vn97::kTokenizerAddBos |
                vn97::kTokenizerAddText |
                vn97::kTokenizerAddEos,
            ids,
            9,
            &count) ==
        vn97::TokenizerStatus::kOk);

    assert(count == 6);
    assert(ids[0] == 1);
    assert(ids[1] == 3);
    assert(ids[2] == 265);
    assert(ids[3] == 264);
    assert(
        ids[4] ==
        8 +
        static_cast<
            std::uint32_t
        >('x'));
    assert(ids[5] == 2);

    std::uint8_t decoded[
        sizeof(input)
    ] = {};

    std::size_t decoded_size = 0;

    assert(
        vn97::TokenizerDecodeBytes(
            view,
            ids,
            count,
            true,
            decoded,
            sizeof(decoded),
            &decoded_size) ==
        vn97::TokenizerStatus::kOk);

    assert(
        decoded_size ==
        sizeof(input));

    assert(
        std::memcmp(
            decoded,
            input,
            sizeof(input)) == 0);

    std::size_t required = 0;
    std::uint8_t tiny[1] = {};

    assert(
        vn97::TokenizerDecodeBytes(
            view,
            ids,
            count,
            true,
            tiny,
            sizeof(tiny),
            &required) ==
        vn97::TokenizerStatus::kOutputTooSmall);

    assert(
        required ==
        sizeof(input));

    auto duplicate =
        BuildPackage(
            {"ab", "abc", "ab"});

    assert(
        vn97::ParseTokenizer(
            duplicate.data(),
            duplicate.size(),
            &view) ==
        vn97::TokenizerStatus::kDuplicateToken);

    auto bad_bucket = blob;

    const std::size_t offset_bytes =
        (3 + 1) * 4;
    const std::size_t bucket_start_bytes =
        257 * 4;
    const std::size_t bucket_index_base =
        24 +
        offset_bytes +
        bucket_start_bytes;

    bad_bucket[
        bucket_index_base + 0
    ] = 2;
    bad_bucket[
        bucket_index_base + 4
    ] = 0;

    assert(
        vn97::ParseTokenizer(
            bad_bucket.data(),
            bad_bucket.size(),
            &view) ==
        vn97::TokenizerStatus::kInvalidLayout);

    return 0;
}
