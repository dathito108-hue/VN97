VN97 bootstrap asset slot

A production release APK MUST place exactly these three public, non-empty files
in this directory:

- model.vn97cap1
- model.vn97sig1
- publisher.ed25519

M19A production release rules:

- model.vn97cap1 must be a safe regular VN97CAP1 file, 96 bytes..512 MiB;
- model.vn97sig1 must be a safe regular file, 1..16384 bytes;
- publisher.ed25519 must be the exact 32-byte raw Ed25519 public key;
- no private key, keystore, password, script, plugin, or extra payload is
  permitted in this directory;
- release signing material is supplied outside the repository through
  VN97_RELEASE_KEYSTORE, VN97_RELEASE_STORE_PASSWORD, VN97_RELEASE_KEY_ALIAS
  and VN97_RELEASE_KEY_PASSWORD;
- the release keystore itself must remain outside the repository.

For release builds Gradle generates:

assets/vn97-release/release.vn97rel1

VN97REL1 binds the application/version identity to the SHA-256 and size of the
three public bootstrap assets. Runtime activation compares the trusted
provisioning result to that manifest. APK upgrades therefore cannot silently
continue using an older active model when the new release carries a different
canonical package.

The release build is zero-provision. On first launch or package replacement,
VN97 verifies/activates the bundled package through the existing canonical
trust + compatibility + transactional activation pipeline.

Debug builds keep the developer import path. Debug-only provisioning is not
part of the end-user installation contract.

Do not put production private signing material in this repository.
