# G0.5 → G0.6 Android packaging checks

The Android reader accepts only float16/float32 recurrent state. The Python
compiler previously copied any dtype (including missing dtype as `"None"`) into
a descriptor. It also accepted boolean values in integer fields and did not
check the state byte budget or the valid-length range it normalized for Android.
These conditions now fail before writing a descriptor.

The shared G0.5 inventory verifier now rejects duplicate JSON keys, non-finite
JSON, symlinked manifests, boolean file sizes, and filenames that can escape the
bundle or contain control characters. File sizes and SHA-256 are still checked;
valid manifest/runtime identities and activation restrictions remain unchanged.
The exporter re-exports the shared constants and verifier at its existing import
path. The contract module itself has no Torch/ONNX dependency.

Run `python tests/host_mamba2_runtime_contract.py` for eight regression tests
(with multiple malformed-input cases). They load the actual packaging modules
without the training package facade and use tiny inventory files, not ONNX
inference models. This proves descriptor/inventory validation only, not numeric
parity, production activation, memory pressure, or phone latency.

## Recovery status at this checkpoint

Run 36406947463 exported candidate ONNX bundles but failed its exact historical
manifest/graph-size selection gate. Repeating the same exporter combinations
will not address that mismatch. The recovered G0.3 capsule in run 36367075535
still requires full payload verification and the established parity/activation
evidence before production use. This patch neither substitutes a candidate for
the canonical artifact nor authorizes production. Graph export, packaging,
numeric parity, and Android execution must remain separately evidenced.
