# M10S — Zero-Provision Turnkey APK Contract

M10S locks the end-user installation contract for VN97:

> Install one APK. The user must not install or import a model, tokenizer,
> signature, publisher key, Python environment, Termux package, native runtime,
> or any other VN97 dependency after installation.

Only normal user interaction choices remain outside the package, such as
permissions, approval of external actions, avatar/voice preferences and other
presentation settings.

## Release vs developer build

VN97 now has two intentional Android profiles:

- **debug/developer** — manual trusted-model import remains available for
  development and recovery work;
- **release/turnkey** — a complete signed VN97 bootstrap is mandatory and
  manual model provisioning is removed from the normal UI.

The release build exposes `BuildConfig.VN97_TURNKEY_REQUIRED=true`.
The debug build exposes it as `false`.

## Mandatory intelligence payload

A release APK must contain these non-empty assets:

- `assets/vn97-bootstrap/model.vn97cap1`;
- `assets/vn97-bootstrap/model.vn97sig1`;
- `assets/vn97-bootstrap/publisher.ed25519`.

The Gradle `preReleaseBuild` path depends on
`verifyTurnkeyBootstrap`. A release build fails before packaging if any
required asset is missing or empty.

This means an apparently successful APK cannot silently ship in
`MODEL_REQUIRED` state.

## First-launch behavior

A release first launch performs:

`APK asset -> VN97CAP1 parse/stage -> Ed25519 trust verification ->
compatibility planning -> transactional activation -> VN97MI1 native
validation -> VN97INV1 -> activated model -> sovereign memory creation ->
assistant session`.

The same trust pipeline used for imported models is preserved. Bundling a model
does not create a bypass.

If the bundled bootstrap is absent, incomplete, corrupt, untrusted,
incompatible or cannot be activated, the release fails closed rather than
falling back to a user-facing manual provisioning workflow.

## End-user UX

In a turnkey release:

- the IMPORT VN97 MODEL button is hidden;
- package/signature/publisher-key selection is not available;
- the app starts by preparing and verifying its bundled intelligence;
- chat becomes enabled only after the activated model is opened successfully.

The developer build keeps manual provisioning because it is useful during
training and integration, but it is not part of the final installation
contract.

## What still must be bundled before the final APK is actually complete

M10S enforces the packaging contract; it does not invent a production-quality
checkpoint. A final installable VN97 APK still requires a real release-qualified
VN97CAMP2 winner to be converted and signed through M10N into the three
bootstrap assets above.

The complete production chain is now:

`training corpus -> M10P/M10Q/M10R campaign -> sealed release winner ->
VN97CK1 + VN97TK1 -> M10N signed bootstrap -> mandatory release assets ->
release APK -> automatic first-launch activation`.

## Architecture boundary

M10S does not introduce Transformer/LLaMA, an alternate AI backend, cloud
inference or a second model path. It packages only the canonical VN97
Code-1 -> Code-2 production model and its existing native runtime.
