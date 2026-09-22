from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import struct
from typing import Iterable, Sequence


MAGIC = b"VN97TK1\0"
VERSION = 1
CONTROL_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<text>",
    "<audio>",
    "<vision>",
    "<tool>",
    "<memory>",
)
CONTROL_COUNT = len(CONTROL_TOKENS)
BYTE_BASE = CONTROL_COUNT
BYTE_COUNT = 256
LEARNED_BASE = BYTE_BASE + BYTE_COUNT
_HEADER = struct.Struct("<8sIIII")
_U32 = struct.Struct("<I")


@dataclass(frozen=True)
class VN97TokenizerPackage:
    learned_tokens: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        seen: set[bytes] = set()
        for token in self.learned_tokens:
            if not isinstance(token, bytes):
                raise TypeError("learned tokens must be bytes")
            if len(token) < 2:
                raise ValueError("learned tokens must contain at least two bytes")
            if token in seen:
                raise ValueError("learned tokens must be unique")
            seen.add(token)

    @property
    def vocab_size(self) -> int:
        return LEARNED_BASE + len(self.learned_tokens)

    def to_bytes(self) -> bytes:
        offsets = [0]
        token_data = bytearray()
        for token in self.learned_tokens:
            token_data.extend(token)
            if len(token_data) > 0xFFFFFFFF:
                raise ValueError("VN97TK1 token payload exceeds uint32 offsets")
            offsets.append(len(token_data))
        header = _HEADER.pack(
            MAGIC,
            VERSION,
            CONTROL_COUNT,
            BYTE_BASE,
            len(self.learned_tokens),
        )
        table = b"".join(_U32.pack(offset) for offset in offsets)
        return header + table + token_data

    @classmethod
    def from_bytes(cls, blob: bytes) -> "VN97TokenizerPackage":
        if len(blob) < _HEADER.size:
            raise ValueError("tokenizer package is shorter than the header")
        magic, version, control_count, byte_base, learned_count = (
            _HEADER.unpack_from(blob)
        )
        if magic != MAGIC:
            raise ValueError("bad VN97TK1 magic")
        if version != VERSION:
            raise ValueError(
                f"unsupported VN97TK1 version: {version}"
            )
        if (
            control_count != CONTROL_COUNT
            or byte_base != BYTE_BASE
        ):
            raise ValueError(
                "incompatible VN97TK1 control/byte layout"
            )

        table_size = (learned_count + 1) * _U32.size
        table_end = _HEADER.size + table_size
        if table_end > len(blob):
            raise ValueError(
                "truncated learned-token offset table"
            )
        offsets = [
            _U32.unpack_from(
                blob,
                _HEADER.size + i * _U32.size,
            )[0]
            for i in range(learned_count + 1)
        ]
        if offsets[0] != 0:
            raise ValueError(
                "VN97TK1 first learned-token offset must be zero"
            )
        token_data = blob[table_end:]
        if offsets[-1] != len(token_data):
            raise ValueError(
                "VN97TK1 final offset does not match payload size"
            )

        learned: list[bytes] = []
        previous = 0
        for offset in offsets[1:]:
            if offset < previous or offset > len(token_data):
                raise ValueError(
                    "invalid VN97TK1 learned-token offset"
                )
            token = bytes(token_data[previous:offset])
            if len(token) < 2:
                raise ValueError(
                    "learned tokens must contain at least two bytes"
                )
            learned.append(token)
            previous = offset
        return cls(tuple(learned))


class VN97Tokenizer:
    """Deterministic byte-lossless tokenizer with learned multi-byte tokens."""

    pad_id = 0
    bos_id = 1
    eos_id = 2
    text_id = 3
    audio_id = 4
    vision_id = 5
    tool_id = 6
    memory_id = 7

    def __init__(
        self,
        package: VN97TokenizerPackage | None = None,
    ) -> None:
        self.package = package or VN97TokenizerPackage()
        buckets: dict[
            int,
            list[tuple[bytes, int]],
        ] = {}
        for index, token in enumerate(
            self.package.learned_tokens
        ):
            token_id = LEARNED_BASE + index
            buckets.setdefault(
                token[0], []
            ).append((token, token_id))
        for values in buckets.values():
            values.sort(
                key=lambda item: (
                    -len(item[0]),
                    item[1],
                )
            )
        self._buckets = buckets

    @property
    def vocab_size(self) -> int:
        return self.package.vocab_size

    def encode_bytes(
        self,
        data: bytes,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        add_text_tag: bool = False,
    ) -> list[int]:
        output: list[int] = []
        if add_bos:
            output.append(self.bos_id)
        if add_text_tag:
            output.append(self.text_id)

        offset = 0
        while offset < len(data):
            matched_id: int | None = None
            matched_size = 0
            for token, token_id in self._buckets.get(
                data[offset], ()
            ):
                end = offset + len(token)
                if (
                    end <= len(data)
                    and data[offset:end] == token
                ):
                    matched_id = token_id
                    matched_size = len(token)
                    break
            if matched_id is None:
                output.append(
                    BYTE_BASE + data[offset]
                )
                offset += 1
            else:
                output.append(matched_id)
                offset += matched_size

        if add_eos:
            output.append(self.eos_id)
        return output

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        add_text_tag: bool = False,
    ) -> list[int]:
        return self.encode_bytes(
            text.encode("utf-8"),
            add_bos=add_bos,
            add_eos=add_eos,
            add_text_tag=add_text_tag,
        )

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_control: bool = True,
    ) -> bytes:
        output = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    "token IDs must be integers"
                )
            if 0 <= token_id < CONTROL_COUNT:
                if skip_control:
                    continue
                raise ValueError(
                    "control tokens do not map to transport bytes"
                )
            if (
                BYTE_BASE
                <= token_id
                < LEARNED_BASE
            ):
                output.append(
                    token_id - BYTE_BASE
                )
                continue
            learned_index = (
                token_id - LEARNED_BASE
            )
            if (
                0 <= learned_index
                < len(
                    self.package.learned_tokens
                )
            ):
                output.extend(
                    self.package.learned_tokens[
                        learned_index
                    ]
                )
                continue
            raise ValueError(
                f"token ID out of range: {token_id}"
            )
        return bytes(output)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_control: bool = True,
        errors: str = "strict",
    ) -> str:
        return self.decode_bytes(
            token_ids,
            skip_control=skip_control,
        ).decode(
            "utf-8",
            errors=errors,
        )


def learn_byte_bpe(
    corpus: Iterable[str | bytes],
    *,
    max_learned_tokens: int,
    min_pair_count: int = 2,
) -> VN97TokenizerPackage:
    """Learn deterministic byte-level pair merges without external dependencies."""
    if max_learned_tokens < 0:
        raise ValueError(
            "max_learned_tokens must be non-negative"
        )
    if min_pair_count < 2:
        raise ValueError(
            "min_pair_count must be at least 2"
        )

    sequences: list[list[bytes]] = []
    for sample in corpus:
        data = (
            sample.encode("utf-8")
            if isinstance(sample, str)
            else bytes(sample)
        )
        if data:
            sequences.append(
                [
                    bytes((value,))
                    for value in data
                ]
            )

    learned: list[bytes] = []
    learned_set: set[bytes] = set()
    while len(learned) < max_learned_tokens:
        counts: Counter[
            tuple[bytes, bytes]
        ] = Counter()
        for sequence in sequences:
            counts.update(
                zip(
                    sequence,
                    sequence[1:],
                )
            )
        if not counts:
            break

        ranked = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0][0] + item[0][1],
                item[0][0],
                item[0][1],
            ),
        )
        pair: (
            tuple[bytes, bytes]
            | None
        ) = None
        merged = b""
        for candidate, count in ranked:
            if count < min_pair_count:
                break
            candidate_bytes = (
                candidate[0]
                + candidate[1]
            )
            if (
                candidate_bytes
                not in learned_set
            ):
                pair = candidate
                merged = candidate_bytes
                break
        if pair is None:
            break

        learned.append(merged)
        learned_set.add(merged)

        rewritten: list[
            list[bytes]
        ] = []
        for sequence in sequences:
            next_sequence: list[
                bytes
            ] = []
            i = 0
            while i < len(sequence):
                if (
                    i + 1
                    < len(sequence)
                    and sequence[i]
                    == pair[0]
                    and sequence[i + 1]
                    == pair[1]
                ):
                    next_sequence.append(
                        merged
                    )
                    i += 2
                else:
                    next_sequence.append(
                        sequence[i]
                    )
                    i += 1
            rewritten.append(
                next_sequence
            )
        sequences = rewritten

    return VN97TokenizerPackage(
        tuple(learned)
    )
