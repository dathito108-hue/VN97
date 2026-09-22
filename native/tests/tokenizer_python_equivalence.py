from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from vn97.tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)


U8P = ctypes.POINTER(
    ctypes.c_uint8
)
U32P = ctypes.POINTER(
    ctypes.c_uint32
)


def main() -> None:
    lib = ctypes.CDLL(
        str(
            Path(
                sys.argv[1]
            ).resolve()
        )
    )
    lib.vn97_tokenizer_encode_bytes.argtypes = [
        U8P,
        ctypes.c_size_t,
        U8P,
        ctypes.c_size_t,
        ctypes.c_uint32,
        U32P,
        ctypes.c_size_t,
        ctypes.POINTER(
            ctypes.c_size_t
        ),
    ]
    lib.vn97_tokenizer_encode_bytes.restype = (
        ctypes.c_int
    )
    lib.vn97_tokenizer_decode_bytes.argtypes = [
        U8P,
        ctypes.c_size_t,
        U32P,
        ctypes.c_size_t,
        ctypes.c_int,
        U8P,
        ctypes.c_size_t,
        ctypes.POINTER(
            ctypes.c_size_t
        ),
    ]
    lib.vn97_tokenizer_decode_bytes.restype = (
        ctypes.c_int
    )

    package = VN97TokenizerPackage(
        (
            b"ab",
            b"abc",
            " Việt".encode("utf-8"),
            "🤖".encode("utf-8"),
        )
    )
    tokenizer = VN97Tokenizer(
        package
    )
    blob = package.to_bytes()
    data = (
        "abcab VN97 Việt 🤖"
        .encode("utf-8")
    )
    expected = tokenizer.encode_bytes(
        data,
        add_bos=True,
        add_text_tag=True,
        add_eos=True,
    )

    blob_array = (
        ctypes.c_uint8
        * len(blob)
    ).from_buffer_copy(blob)
    data_array = (
        ctypes.c_uint8
        * len(data)
    ).from_buffer_copy(data)
    output = (
        ctypes.c_uint32
        * (len(data) + 3)
    )()
    count = ctypes.c_size_t()

    status = (
        lib.vn97_tokenizer_encode_bytes(
            blob_array,
            len(blob),
            data_array,
            len(data),
            0b111,
            output,
            len(output),
            ctypes.byref(count),
        )
    )
    assert status == 0

    actual = list(
        output[: count.value]
    )
    assert actual == expected

    id_array = (
        ctypes.c_uint32
        * len(actual)
    )(*actual)
    decoded = (
        ctypes.c_uint8
        * len(data)
    )()
    decoded_size = ctypes.c_size_t()

    status = (
        lib.vn97_tokenizer_decode_bytes(
            blob_array,
            len(blob),
            id_array,
            len(actual),
            1,
            decoded,
            len(decoded),
            ctypes.byref(
                decoded_size
            ),
        )
    )
    assert status == 0
    assert bytes(
        decoded[
            : decoded_size.value
        ]
    ) == data


if __name__ == "__main__":
    main()
