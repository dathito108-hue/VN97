VN97 bootstrap asset slot

A turnkey APK may place exactly these three files in this directory:

- model.vn97cap1
- model.vn97sig1
- publisher.ed25519

They are optional. If none of the three are present, VN97 remains MODEL_REQUIRED and the app offers
manual trusted-model import. If only some are present, cold start fails closed.

Do not place native test fixtures here and do not mark a model active without the M10D/M10E/M10F/
M10G trust + compatibility + transactional activation pipeline.
