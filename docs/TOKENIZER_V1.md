# VN97TK1 tokenizer format

VN97TK1 is the canonical M3A tokenizer package. It is designed so text tokenization never loses
information and the native runtime does not depend on an external tokenizer library.

## Token IDs

The first eight IDs are fixed controls:

    0  <pad>
    1  <bos>
    2  <eos>
    3  <text>
    4  <audio>
    5  <vision>
    6  <tool>
    7  <memory>

Raw byte tokens occupy IDs 8 through 263:

    token_id = 8 + byte_value

Every possible byte therefore has a direct representation. UTF-8 text is encoded over these
bytes, but encode_bytes/decode_bytes can also round-trip arbitrary binary data.

Learned multi-byte tokens begin at ID 264. A learned token must contain at least two bytes and
learned token byte strings must be unique.

## Binary layout

All integers are little-endian uint32.

Header, 24 bytes:

    offset 0   magic[8] = "VN97TK1\0"
    offset 8   version = 1
    offset 12  control_count = 8
    offset 16  byte_base = 8
    offset 20  learned_count = V

The header is followed by:

    learned_offsets[V + 1]
    bucket_starts[257]
    bucket_indices[V]
    learned_token_bytes[...]

learned_offsets are relative to learned_token_bytes. The first offset is zero and the final
offset equals the learned-token payload size.

bucket_starts partitions bucket_indices by first byte. bucket_indices is a permutation of
0..V-1. Within each first-byte bucket entries are ordered by descending token byte length, then
ascending learned-token index. This means the native encoder can take the first matching
candidate and obtain deterministic longest-prefix semantics without scanning unrelated first
bytes.

## Encode semantics

For each byte position:

1. select the bucket for the current first byte;
2. test candidates in serialized longest-first order;
3. emit the first learned token that matches;
4. if none matches, emit the canonical raw-byte ID;
5. advance by the emitted token byte length.

Equal-length ties resolve to the lower learned-token ID.

Optional BOS, TEXT and EOS controls can be added around text input. Control tokens never
replace transport bytes.

## Decode semantics

Byte IDs map directly to one raw byte. Learned IDs append their stored byte sequence. Control
IDs are skipped when skip_control is true; otherwise decode rejects controls because they do
not represent transport bytes.

Therefore:

    decode_bytes(encode_bytes(data)) == data

for every byte sequence.

## Vocabulary learning

learn_byte_bpe is a deterministic dependency-free reference builder. It learns pair merges
from byte sequences and produces a VN97TokenizerPackage. VN97TK1 runtime semantics do not
require BPE specifically: any producer may supply unique learned byte sequences as long as the
package invariants are satisfied.

## Native boundary

The C++17 runtime exposes TokenizerView plus:

- ParseTokenizer
- TokenizerEncodeBytes
- TokenizerDecodeBytes

and C-linkage wrappers:

- vn97_tokenizer_encode_bytes
- vn97_tokenizer_decode_bytes

A parsed TokenizerView references the package bytes without copying token payloads. The
serialized bucket index keeps per-position matching scoped to the current first-byte bucket.
