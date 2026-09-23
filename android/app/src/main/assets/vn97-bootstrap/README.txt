VN97 bootstrap asset slot

A production release APK MUST place exactly these three non-empty files in this directory:

- model.vn97cap1
- model.vn97sig1
- publisher.ed25519

The release build is zero-provision: Gradle rejects a release APK when any of the three assets is
missing or empty. On first launch the app verifies and activates the bundled package through the
existing M10D/M10E/M10F/M10G trust + compatibility + transactional activation pipeline.

Debug builds keep the developer import path so VN97 development can continue before a production
checkpoint is ready. Debug-only manual provisioning is not part of the end-user install flow.

Do not place native test fixtures here and do not mark a model active without the canonical trust
pipeline.
