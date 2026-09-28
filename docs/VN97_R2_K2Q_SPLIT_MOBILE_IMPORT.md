# VN97 R2-K2Q split mobile import

Some Android browsers do not reliably start a single 5.1 GiB Kaggle download.

The K2Q evidence APK accepts either the original recurrent-8.onnx.data file or contiguous split parts named recurrent-8.onnx.data.part000, part001, and so on.

When split parts are selected, the APK streams them directly into one private recurrent-8.onnx.data file. The existing G0.6 verifier then checks final byte length and SHA-256 before accepting the runtime.

After K2Q recovery PASS, run:

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2q_split_android_data.sh

Output is under /kaggle/working/K2Q-PARTS. The default 512 MiB chunk size produces 11 parts.

Download runtime.vn97m2g06.json, recurrent-8.onnx, every partNNN file, and K2Q_SPLIT_SHA256SUMS.txt. In the K2Q Evidence APK, select the runtime JSON, ONNX graph, and all split parts together.
