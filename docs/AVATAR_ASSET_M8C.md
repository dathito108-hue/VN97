# M8C — VN97AV1 Native Avatar Asset + Rig Contract

M8C completes the core interactive-avatar architecture by defining a sovereign, bounded asset
format and a semantic rig-pose layer.

## VN97AV1 binary layout

VN97AV1 is little-endian and begins with an 80-byte fixed header.

- magic: `VN97AV1\0`;
- version: 1;
- header size: 80;
- flags/reserved fields: zero;
- vertex count;
- triangle index count;
- joint count;
- vertex stride: 32 bytes;
- joint stride: 48 bytes;
- exact vertex/index/joint offsets;
- exact total blob size;
- payload CRC32;
- header CRC32 over bytes 0..75.

Sections must be contiguous and the actual blob length must equal the declared total size.

CRC32 detects corruption only. It is not a signature or provenance/authentication mechanism.

## Geometry bounds

The parser rejects assets outside these reference limits:

- vertices: 3..200,000;
- indices: 3..600,000 and divisible by 3;
- joints: 0..128.

Every vertex stores position F32[3], normal F32[3], four joint IDs and four U8 skin weights.
Position/normal values must be finite and bounded. Normal squared length must remain in a
reasonable normalized range.

Every index must reference an existing vertex.

For rigged assets, four skin weights per vertex must sum exactly to 255 and every nonzero-weight
joint ID must exist. For unrigged assets the weight sum must be zero.

## Rig records

Each 48-byte joint contains:

- parent I32;
- translation F32[3];
- quaternion XYZW F32[4];
- scale F32[3];
- reserved U32 = 0.

Parents are either -1 for a root or refer to an earlier joint, making the serialized hierarchy
topologically acyclic. At least one root is required for a rigged asset.

Translations must be finite/bounded, quaternion length squared must remain near one, and scale
components must be finite, positive and bounded.

## Native API

`libvn97_avatar_asset.a` exposes:

- `ParseAvatarAsset`;
- `ReadAvatarVertex`;
- `ReadAvatarIndex`;
- `ReadAvatarJoint`;
- C ABI `vn97_avatar_asset_parse`.

The parser validates the complete asset before returning a view.

`AvatarAssetView` references caller-owned bytes and therefore must never outlive that storage.

## Android JNI boundary

The `:avatar` module builds `libvn97_avatar_jni.so`, linked only to the VN97 native asset
validator.

`NativeAvatarAsset.validate()`:

1. enforces a caller-configured size bound (64 MiB default);
2. passes the in-memory byte array to the native validator;
3. maps exact native status codes;
4. returns vertex/index/joint metadata;
5. stores a defensive immutable copy of the validated bytes.

JNI does not open files, load arbitrary dynamic code or perform an external action.

## Semantic rig pose

`AvatarRigBinding` binds semantic slots such as root/head/jaw/arms to validated joint IDs and
requires in-range unique bindings.

`AvatarRigAnimator` converts M8 immutable frame state into bounded semantic controls:

- head pitch/yaw;
- jaw opening;
- left/right arm roll;
- brow raise;
- smile;
- mouth width;
- lip rounding.

Gaze, listening, speaking, approval/error mode and M8 gestures drive these controls. This pose
layer does not change the trusted asset bytes and does not call cognition.

## Verification

The exact M8C native parser snapshot was compiled and executed with:

- C++17;
- `-Wall -Wextra -Werror -pedantic`;
- AddressSanitizer;
- UndefinedBehaviorSanitizer.

Regression coverage includes valid mesh/rig reads, payload corruption, out-of-range indices,
invalid skin weights, invalid parent hierarchy, non-finite geometry and invalid header stride.

The Kotlin rig-pose test compiles with `kotlinc -Werror` and covers binding bounds/uniqueness,
gesture/head/jaw/arm controls and alert/approval facial controls.

A host JNI regression compiles `libvn97_avatar_jni.so`, validates a real VN97AV1 byte blob,
checks metadata and defensive copying, then verifies corruption rejection.

Actual Android GPU rendering of imported VN97AV1 meshes remains a device integration concern;
M8C fixes the validation/rig contract without replacing the procedural M8 renderer by default.
