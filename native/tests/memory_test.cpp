#include "vn97/memory.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

void PushU16(
    std::vector<std::uint8_t>& out,
    std::uint16_t value) {
    out.push_back(
        static_cast<std::uint8_t>(value));
    out.push_back(
        static_cast<std::uint8_t>(value >> 8));
}

void PushU32(
    std::vector<std::uint8_t>& out,
    std::uint32_t value) {
    out.push_back(
        static_cast<std::uint8_t>(value));
    out.push_back(
        static_cast<std::uint8_t>(value >> 8));
    out.push_back(
        static_cast<std::uint8_t>(value >> 16));
    out.push_back(
        static_cast<std::uint8_t>(value >> 24));
}

void PushU64(
    std::vector<std::uint8_t>& out,
    std::uint64_t value) {
    PushU32(
        out,
        static_cast<std::uint32_t>(value));
    PushU32(
        out,
        static_cast<std::uint32_t>(value >> 32));
}

void PushF32(
    std::vector<std::uint8_t>& out,
    float value) {
    std::uint32_t bits = 0;
    std::memcpy(
        &bits,
        &value,
        sizeof(bits));
    PushU32(out, bits);
}

std::uint32_t Crc32(
    const std::uint8_t* data,
    std::size_t size) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask =
                static_cast<std::uint32_t>(-
                    static_cast<std::int32_t>(
                        crc & 1u));
            crc =
                (crc >> 1) ^
                (0xedb88320u & mask);
        }
    }
    return ~crc;
}

std::vector<std::uint8_t> BuildJournal() {
    std::vector<std::uint8_t> journal = {
        'V', 'N', '9', '7',
        'M', 'E', 'M', '1'
    };
    PushU32(journal, 1);
    PushU32(journal, 2);

    std::vector<std::uint8_t> body;
    PushU64(body, 1);
    PushU64(body, 100);
    body.push_back(1);
    body.push_back(0);
    PushU16(body, 0);
    PushF32(body, 0.75f);
    PushU32(body, 4);
    PushU32(body, 5);
    PushU32(body, 2);
    PushU64(body, 0);
    for (int i = 0; i < 32; ++i) {
        body.push_back(0);
    }
    body.insert(
        body.end(),
        {'u', 'n', 'i', 't'});
    body.insert(
        body.end(),
        {'h', 'e', 'l', 'l', 'o'});
    PushF32(body, 1.0f);
    PushF32(body, 0.0f);

    PushU32(
        journal,
        static_cast<std::uint32_t>(
            body.size()));
    PushU32(
        journal,
        Crc32(
            body.data(),
            body.size()));
    journal.insert(
        journal.end(),
        body.begin(),
        body.end());
    return journal;
}

}  // namespace

int main() {
    auto journal = BuildJournal();
    vn97::MemoryScanResult result;

    assert(
        vn97::ScanMemoryJournal(
            journal.data(),
            journal.size(),
            false,
            &result) ==
        vn97::MemoryStatus::kOk);
    assert(result.vector_dim == 2);
    assert(result.record_count == 1);
    assert(result.last_record_id == 1);
    assert(
        result.valid_bytes ==
        journal.size());
    assert(!result.tail_truncated);

    auto torn = journal;
    torn.insert(
        torn.end(),
        {0x40, 0x00, 0x00});
    assert(
        vn97::ScanMemoryJournal(
            torn.data(),
            torn.size(),
            false,
            &result) ==
        vn97::MemoryStatus::kTruncatedTail);
    assert(
        vn97::ScanMemoryJournal(
            torn.data(),
            torn.size(),
            true,
            &result) ==
        vn97::MemoryStatus::kOk);
    assert(result.tail_truncated);
    assert(
        result.valid_bytes ==
        journal.size());

    auto corrupt = journal;
    corrupt.back() ^= 0x01;
    assert(
        vn97::ScanMemoryJournal(
            corrupt.data(),
            corrupt.size(),
            true,
            &result) ==
        vn97::MemoryStatus::kChecksumMismatch);

    auto bad_magic = journal;
    bad_magic[0] = 'X';
    assert(
        vn97::ScanMemoryJournal(
            bad_magic.data(),
            bad_magic.size(),
            false,
            &result) ==
        vn97::MemoryStatus::kBadMagic);

    return 0;
}
