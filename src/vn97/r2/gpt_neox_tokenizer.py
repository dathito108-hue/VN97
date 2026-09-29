from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
import unicodedata
from typing import Mapping, Sequence

from .mamba2_g03_capsule import (
    TOKENIZER_FILES,
    verify_g03_capsule_structure,
)
from .mamba2_transfer import (
    MAMBA2_27B_VOCAB_SIZE,
    VN97_MAMBA2_SOURCE_TOKENIZER_ID,
    VN97_MAMBA2_SOURCE_TOKENIZER_REVISION,
)


VN97_MAMBA2_G08_TOKENIZER_SCHEMA = "VN97M2G08TOK1"
VN97_MAMBA2_G08_TOKENIZER_FILENAME = "tokenizer.vn97m2g08.json"
GPT_NEOX_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| "
    r"?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
RUNTIME_LOGITS_SIZE = 50_288


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def bytes_to_unicode() -> tuple[dict[int, str], dict[str, int]]:
    base = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = list(base)
    extra = 0
    for value in range(256):
        if value not in base:
            base.append(value)
            codepoints.append(256 + extra)
            extra += 1
    encoder = {
        byte: chr(codepoint)
        for byte, codepoint in zip(base, codepoints, strict=True)
    }
    decoder = {character: byte for byte, character in encoder.items()}
    if len(encoder) != 256 or len(decoder) != 256:
        raise AssertionError("GPT-NeoX byte/unicode table must be bijective")
    return encoder, decoder


def _is_letter(value: str) -> bool:
    return unicodedata.category(value).startswith("L")


def _is_number(value: str) -> bool:
    return unicodedata.category(value).startswith("N")


def _is_other_nonspace(value: str) -> bool:
    return (
        not value.isspace()
        and not _is_letter(value)
        and not _is_number(value)
    )


def gpt_neox_pretokenize(text: str) -> tuple[str, ...]:
    """Dependency-free equivalent of the canonical GPT-2/NeoX regex.

    The final whitespace branch reproduces the whitespace negative-lookahead
    behavior: before a non-whitespace token, a multi-whitespace run emits all
    but its final character first, allowing a final ASCII space to attach to
    the following token.
    """

    contractions = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")
    output: list[str] = []
    index = 0
    size = len(text)

    while index < size:
        contraction = next(
            (
                value
                for value in contractions
                if text.startswith(value, index)
            ),
            None,
        )
        if contraction is not None:
            output.append(contraction)
            index += len(contraction)
            continue

        leading_space = text[index] == " " and index + 1 < size
        start = index + 1 if leading_space else index

        if start < size and _is_letter(text[start]):
            end = start + 1
            while end < size and _is_letter(text[end]):
                end += 1
            output.append(text[index:end])
            index = end
            continue

        if start < size and _is_number(text[start]):
            end = start + 1
            while end < size and _is_number(text[end]):
                end += 1
            output.append(text[index:end])
            index = end
            continue

        if start < size and _is_other_nonspace(text[start]):
            end = start + 1
            while end < size and _is_other_nonspace(text[end]):
                end += 1
            output.append(text[index:end])
            index = end
            continue

        if text[index].isspace():
            end = index + 1
            while end < size and text[end].isspace():
                end += 1
            if end == size:
                output.append(text[index:end])
                index = end
                continue
            run = end - index
            if run >= 2:
                output.append(text[index : end - 1])
                index = end - 1
            else:
                output.append(text[index:end])
                index = end
            continue

        raise AssertionError(
            f"GPT-NeoX pretokenizer made no progress at index {index}"
        )

    return tuple(output)


def _read_vocab(path: Path) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("GPT-NeoX vocab.json must be non-empty object")
    vocab: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in raw.items():
        if (
            not isinstance(token, str)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id in seen_ids
        ):
            raise ValueError("GPT-NeoX vocabulary entry is invalid")
        vocab[token] = token_id
        seen_ids.add(token_id)
    if seen_ids != set(range(max(seen_ids) + 1)):
        raise ValueError("GPT-NeoX vocabulary IDs must be contiguous")
    return vocab


def _read_merges(path: Path) -> tuple[tuple[str, str], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in lines:
        if not line or line.startswith("#version:"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("GPT-NeoX merges line must contain two symbols")
        pair = (parts[0], parts[1])
        if pair in seen:
            raise ValueError("GPT-NeoX merges contain duplicate pair")
        seen.add(pair)
        output.append(pair)
    if not output:
        raise ValueError("GPT-NeoX merges.txt must contain ranked merges")
    return tuple(output)


@dataclass
class GptNeoXBpeTokenizer:
    vocab: dict[str, int]
    merges: tuple[tuple[str, str], ...]
    eos_token: str = "<|endoftext|>"

    added_tokens: dict[str, int] | None = None
    normalize_nfc: bool = False

    def __post_init__(self) -> None:
        if self.eos_token not in self.vocab:
            raise ValueError("GPT-NeoX EOS token missing from vocabulary")
        self.added_tokens = dict(self.added_tokens or {self.eos_token: self.vocab[self.eos_token]})
        if any(not token or self.vocab.get(token) != token_id for token, token_id in self.added_tokens.items()):
            raise ValueError("GPT-NeoX added token differs from vocabulary")
        self._added_ids = set(self.added_tokens.values())
        self._added_pattern = re.compile("(" + "|".join(re.escape(token) for token in sorted(self.added_tokens, key=len, reverse=True)) + ")")
        self._inverse = {value: key for key, value in self.vocab.items()}
        if len(self._inverse) != len(self.vocab):
            raise ValueError("GPT-NeoX vocabulary IDs are not unique")
        self._ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }
        self._byte_encoder, self._byte_decoder = bytes_to_unicode()
        self._cache: dict[str, tuple[str, ...]] = {}
        if self.eos_token not in self.vocab:
            raise ValueError("GPT-NeoX EOS token missing from vocabulary")

    @classmethod
    def from_files(
        cls,
        vocab_file: Path,
        merges_file: Path,
        *,
        eos_token: str = "<|endoftext|>",
    ) -> "GptNeoXBpeTokenizer":
        added = None
        normalize_nfc = False
        configuration = vocab_file.parent / "tokenizer.json"
        if configuration.is_file():
            metadata = json.loads(configuration.read_text(encoding="utf-8"))
            if metadata.get("normalizer") != {"type": "NFC"}:
                raise ValueError("GPT-NeoX requires the pinned NFC normalizer")
            normalize_nfc = True
            added = {}
            for entry in metadata["added_tokens"]:
                if any(entry.get(flag) is not False for flag in ("single_word", "lstrip", "rstrip")):
                    raise ValueError("unsupported GPT-NeoX added-token matching flags")
                if entry["content"] in added:
                    raise ValueError("duplicate GPT-NeoX added token")
                added[entry["content"]] = entry["id"]
        return cls(
            _read_vocab(vocab_file),
            _read_merges(merges_file),
            eos_token=eos_token,
            added_tokens=added,
            normalize_nfc=normalize_nfc,
        )

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def token_id_space(self) -> int:
        return max(self.vocab.values()) + 1

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        if len(word) <= 1:
            self._cache[token] = word
            return word

        while len(word) > 1:
            pairs = {
                (word[index], word[index + 1])
                for index in range(len(word) - 1)
            }
            ranked = [
                (self._ranks[pair], pair)
                for pair in pairs
                if pair in self._ranks
            ]
            if not ranked:
                break
            _, selected = min(ranked, key=lambda item: item[0])
            first, second = selected
            merged: list[str] = []
            index = 0
            while index < len(word):
                if (
                    index + 1 < len(word)
                    and word[index] == first
                    and word[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)

        self._cache[token] = word
        return word

    def _encode_ordinary(self, text: str) -> list[int]:
        output: list[int] = []
        for piece in gpt_neox_pretokenize(text):
            mapped = "".join(
                self._byte_encoder[value]
                for value in piece.encode("utf-8")
            )
            for bpe_token in self._bpe(mapped):
                token_id = self.vocab.get(bpe_token)
                if token_id is None:
                    raise ValueError(
                        "GPT-NeoX BPE produced token absent from vocabulary"
                    )
                output.append(token_id)
        return output

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        output: list[int] = []
        if self.normalize_nfc:
            text = unicodedata.normalize("NFC", text)
        for part in self._added_pattern.split(text):
            if part in self.added_tokens:
                output.append(self.added_tokens[part])
            elif part:
                output.extend(self._encode_ordinary(part))
        return output

    def decode_bytes(
        self,
        token_ids: Sequence[int],
        *,
        skip_eos: bool = False,
    ) -> bytes:
        output = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("GPT-NeoX token IDs must be integers")
            token = self._inverse.get(token_id)
            if token is None:
                raise ValueError(
                    f"GPT-NeoX token ID out of vocabulary: {token_id}"
                )
            if token_id in self._added_ids:
                if token == self.eos_token and skip_eos:
                    continue
                output.extend(token.encode("utf-8"))
                continue
            for character in token:
                byte = self._byte_decoder.get(character)
                if byte is None:
                    raise ValueError(
                        "GPT-NeoX vocabulary token is outside byte decoder"
                    )
                output.append(byte)
        return bytes(output)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_eos: bool = False,
        errors: str = "strict",
    ) -> str:
        return self.decode_bytes(
            token_ids,
            skip_eos=skip_eos,
        ).decode("utf-8", errors=errors)


def _strict_tokenizer_config(root: Path) -> tuple[str, str, str, bool]:
    config = json.loads(
        (root / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    special = json.loads(
        (root / "special_tokens_map.json").read_text(encoding="utf-8")
    )
    if not isinstance(config, Mapping) or not isinstance(special, Mapping):
        raise ValueError("GPT-NeoX tokenizer metadata must be objects")
    if config.get("tokenizer_class") != "GPTNeoXTokenizer":
        raise ValueError("source tokenizer class is not GPTNeoXTokenizer")
    add_prefix_space = config.get("add_prefix_space")
    if add_prefix_space is not False:
        raise ValueError("pinned GPT-NeoX tokenizer must not add prefix space")
    bos = config.get("bos_token")
    eos = config.get("eos_token")
    unk = config.get("unk_token")
    if not all(isinstance(value, str) for value in (bos, eos, unk)):
        raise ValueError("GPT-NeoX special tokens are missing")
    for field, value in (
        ("bos_token", bos),
        ("eos_token", eos),
        ("unk_token", unk),
    ):
        if special.get(field) != value:
            raise ValueError(
                f"GPT-NeoX {field} differs between tokenizer metadata files"
            )
    return str(bos), str(eos), str(unk), bool(add_prefix_space)


def compile_g08_tokenizer_descriptor(
    tokenizer_root: Path,
    *,
    capsule_id: str,
    expected_token_id_space: int | None = None,
) -> dict[str, object]:
    root = tokenizer_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("G0.8 tokenizer root must be real directory")
    if (
        not isinstance(capsule_id, str)
        or len(capsule_id) != 64
        or any(ch not in "0123456789abcdef" for ch in capsule_id)
    ):
        raise ValueError("G0.8 capsule ID must be lowercase SHA-256")
    for filename in TOKENIZER_FILES:
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"G0.8 tokenizer asset missing: {filename}")

    bos, eos, unk, add_prefix_space = _strict_tokenizer_config(root)
    tokenizer = GptNeoXBpeTokenizer.from_files(
        root / "vocab.json",
        root / "merges.txt",
        eos_token=eos,
    )
    if expected_token_id_space is not None:
        if tokenizer.token_id_space != expected_token_id_space:
            raise ValueError(
                "G0.8 tokenizer ID space differs from source model vocab"
            )
    if tokenizer.token_id_space > RUNTIME_LOGITS_SIZE:
        raise ValueError("G0.8 tokenizer exceeds runtime logits width")

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G08_TOKENIZER_SCHEMA,
        "capsule_id": capsule_id,
        "source_tokenizer_model_id": VN97_MAMBA2_SOURCE_TOKENIZER_ID,
        "source_tokenizer_revision":
            VN97_MAMBA2_SOURCE_TOKENIZER_REVISION,
        "tokenizer_class": "GPTNeoXTokenizer",
        "algorithm": "gpt2_byte_level_bpe_ranked_merges",
        "pretokenizer_pattern": GPT_NEOX_PATTERN,
        "add_prefix_space": add_prefix_space,
        "bos_token": bos,
        "eos_token": eos,
        "unk_token": unk,
        "bos_token_id": tokenizer.vocab[bos],
        "eos_token_id": tokenizer.vocab[eos],
        "unk_token_id": tokenizer.vocab[unk],
        "token_id_space": tokenizer.token_id_space,
        "runtime_logits_size": RUNTIME_LOGITS_SIZE,
        "invalid_padded_token_start": tokenizer.token_id_space,
        "invalid_padded_token_end_exclusive": RUNTIME_LOGITS_SIZE,
        "merge_count": len(tokenizer.merges),
        "assets": {
            filename: {
                "bytes": (root / filename).stat().st_size,
                "sha256": _sha256_file(root / filename),
            }
            for filename in sorted(TOKENIZER_FILES)
        },
        "same_token_ids_required": True,
        "sampling_must_mask_padded_ids": (
            tokenizer.token_id_space < RUNTIME_LOGITS_SIZE
        ),
        "source_runtime_required": False,
        "production_activation_authorized": False,
    }
    descriptor = dict(body)
    descriptor["tokenizer_id"] = _sha256_bytes(
        b"VN97M2G08TOK1\0" + _canonical_json(body)
    )
    return descriptor


def build_g08_from_g03_capsule(
    capsule_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    verified = verify_g03_capsule_structure(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    descriptor = compile_g08_tokenizer_descriptor(
        verified.root / "tokenizer",
        capsule_id=verified.manifest.capsule_id(),
        expected_token_id_space=MAMBA2_27B_VOCAB_SIZE,
    )
    target = output_path or (
        verified.root / "tokenizer" / VN97_MAMBA2_G08_TOKENIZER_FILENAME
    )
    if target.exists() or target.is_symlink():
        raise ValueError("G0.8 tokenizer descriptor output already exists")
    temp = target.with_name(target.name + ".tmp")
    temp.write_bytes(_canonical_json(descriptor) + b"\n")
    temp.replace(target)
    return descriptor
