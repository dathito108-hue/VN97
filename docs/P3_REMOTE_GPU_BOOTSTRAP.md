# P3 — Remote GPU Bootstrap

The P3 CUDA campaign can now be launched from a fresh user-controlled GPU host with
one bootstrap script:

`tools/bootstrap_p3_remote_gpu.sh`

Default working directory:

`$HOME/vn97-p3-gpu`

The bootstrap performs:

1. verify Git, GitHub CLI, Python and sha256sum;
2. require an authenticated GitHub CLI session;
3. clone/fetch `dathito108-hue/VN97`;
4. detach-checkout exact training commit
   `b68428ea9e16f8fd0b42bdd6a76b633456695201`;
5. download the exact P3 corpus artifact from run `35992987070`;
6. verify artifact `SHA256SUMS`;
7. require VN97CORPUS1 identity
   `77e9142054f82b78c8ca0ff108c5e5297d43292841f9fe1ea69c5dc97a7c303f`;
8. launch `tools/run_p3_gpu.sh`;
9. tee the complete campaign output to a durable log;
10. leave VN97P3RUN1 and all winner artifacts under the work directory.

The canonical corpus artifact name is:

`VN97-P3-Corpus-1623f2c187d043918d681a7dc9e9cf025a58d26f`

Its current GitHub artifact retention expires on 2026-10-01T11:26:15Z. A copy should
be pulled onto the GPU host before expiry.

## Host preparation

The remote host needs a working CUDA-enabled PyTorch installation before the bootstrap
starts. The script deliberately does not choose or install a CUDA PyTorch build because
the correct package depends on the rented host driver/runtime.

Minimum checks on the host:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

GitHub CLI also needs access to the repository Actions artifact:

```bash
gh auth login
```

Do not paste access tokens into ChatGPT. Authenticate directly on the remote host.

## Launch

```bash
bash tools/bootstrap_p3_remote_gpu.sh
```

Custom work directory:

```bash
bash tools/bootstrap_p3_remote_gpu.sh /mnt/vn97-p3
```

Optional environment overrides:

```bash
export VN97_REPOSITORY=dathito108-hue/VN97
export VN97_TRAINING_COMMIT=b68428ea9e16f8fd0b42bdd6a76b633456695201
export VN97_P3_CORPUS_RUN_ID=35992987070
export VN97_P3_CORPUS_ARTIFACT=VN97-P3-Corpus-1623f2c187d043918d681a7dc9e9cf025a58d26f
export VN97_CUDA_DEVICE=cuda:0
```

The bootstrap refuses a dirty tracked repository, a non-empty corpus target, a
non-empty output directory, a repository-commit mismatch or a corpus-manifest mismatch.

## ChatGPT remote-computer path

When Remote Desktop Commander exposes an authorized GPU machine to the current chat,
the same bootstrap is suitable for direct execution from ChatGPT. ChatGPT can inspect
the host, verify CUDA, run the bootstrap, monitor the log and read the final
`p3-run.vn97p3run1.json` evidence.

The connection of the plugin itself is not evidence that a CUDA machine is available.
Training starts only after the authorized remote machine exposes a terminal and passes
VN97GPUENV1 preflight.
