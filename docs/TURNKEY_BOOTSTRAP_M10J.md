# M10J — Turnkey Bootstrap Slot + Simplified Model UX

M10J removes the confusing first-launch requirement to understand three provisioning files while
preserving the existing sovereign trust boundary.

## Optional signed bootstrap slot

The APK may contain:

`assets/vn97-bootstrap/model.vn97cap1`
`assets/vn97-bootstrap/model.vn97sig1`
`assets/vn97-bootstrap/publisher.ed25519`

Cold start classifies the slot as:

- ABSENT: none of the required files exist; no model state is fabricated;
- COMPLETE: all three exist; the package is reviewed and activated through the exact M10H
  provisioning session;
- INCOMPLETE: fail closed.

A COMPLETE slot still runs:

`M10D package validation -> M10E Ed25519 verification -> M10F compatibility -> durable publisher
trust -> M10F transaction -> M10G native VN97MI1 validation -> VN97INV1`.

There is no direct asset-to-NativeActivatedModel bypass.

## Why the current APK remains MODEL_REQUIRED

The repository does not yet contain a production VN97MI1 language model with a tokenizer and
assistant-capable trained weights. The only native model-image fixture is a tiny test image and is
not bundled by M10J.

M10J deliberately does not package that fixture as a fake bootstrap model.

## First-launch UX

Technical provisioning controls are collapsed behind one `IMPORT VN97 MODEL` button.

If no bundled model is present, the app says clearly that chat needs a signed VN97 model. Advanced
manual import still uses the same three independently selected VN97CAP1/VN97SIG1/publisher-key
inputs and exact Review -> Trust & Activate path.

If a real bundled bootstrap is added later, first launch verifies and activates it automatically
and hides the advanced provisioning controls.

## Verification

Host contract marker:

`M10J_BOOTSTRAP_ASSET_CONTRACT_PASS`

A full APK build is still required after the UI/bootstrap code is merged. A model-capable turnkey
APK additionally requires an actual trained, tokenizer-bearing VN97MI1 capability; build success
alone does not create intelligence.
