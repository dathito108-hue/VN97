# VN97 R2-K2Q — recover lost Android runtime files

The K2Q Android runtime files are reproducible from the pinned source identity.
They do not depend on the lifetime of the earlier Kaggle session.

The recovery script re-downloads the exact pinned Mamba-2 2.7B source, verifies
its source SHA-256, materializes the same zero-copy G0.3 capsule, exports the
same recurrent-8 G0.5 graph, builds the G0.6 Android descriptor, and refuses to
complete unless all previously measured lineage IDs match exactly.

Expected identities:

- capsule: 8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e
- source weight: 254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be
- G0.5 manifest: ae782f452368b77b0c17dd7f9a2fd284b98bc0b24bc58e4c8e7d82fba046d2ce
- G0.6 runtime: a00feb6b05be1e9219fc9d6bdd6c6ecd9a14fc930202e9689d8c416c6d153762

Use a fresh Kaggle session with at least 13 GiB free in /kaggle/working:

    %cd /kaggle/working
    !git clone https://github.com/dathito108-hue/VN97.git
    %cd /kaggle/working/VN97
    !bash tools/kaggle_r2_k2q_recover_android_files.sh

If VN97 is already cloned:

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2q_recover_android_files.sh

Successful recovery leaves only the three Android import files under:

    /kaggle/working/VN97-K2Q-RECOVERED/

The temporary source and capsule are deleted after the recovered graph and
descriptor have been verified, avoiding a permanent second 5.4 GB copy.
