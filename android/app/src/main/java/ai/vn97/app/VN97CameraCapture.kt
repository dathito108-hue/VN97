package ai.vn97.app

import ai.vn97.runtime.NativePreparedVision
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import java.util.concurrent.atomic.AtomicBoolean

class VN97CameraCapture(
    context: Context,
) {
    private val appContext = context.applicationContext

    @SuppressLint("MissingPermission")
    fun capture(
        onResult: (Result<NativePreparedVision>) -> Unit,
    ) {
        if (
            appContext.checkSelfPermission(Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            onResult(
                Result.failure(
                    SecurityException(
                        "camera permission is not granted"
                    )
                )
            )
            return
        }

        val manager = appContext.getSystemService(CameraManager::class.java)
            ?: run {
                onResult(
                    Result.failure(
                        IllegalStateException(
                            "CameraManager unavailable"
                        )
                    )
                )
                return
            }
        val selection = try {
            selectCamera(manager)
        } catch (exc: Throwable) {
            onResult(Result.failure(exc))
            return
        }

        val thread = HandlerThread("vn97-camera-perception").apply {
            start()
        }
        val handler = Handler(thread.looper)
        val finished = AtomicBoolean(false)
        var device: CameraDevice? = null
        var session: CameraCaptureSession? = null
        val reader = ImageReader.newInstance(
            selection.size.width,
            selection.size.height,
            ImageFormat.JPEG,
            2,
        )

        fun finish(result: Result<NativePreparedVision>) {
            if (!finished.compareAndSet(false, true)) return
            try {
                session?.close()
            } catch (_: RuntimeException) {
                // Best-effort camera cleanup.
            }
            try {
                device?.close()
            } catch (_: RuntimeException) {
                // Best-effort camera cleanup.
            }
            reader.close()
            thread.quitSafely()
            onResult(result)
        }

        reader.setOnImageAvailableListener(
            { source ->
                val image = source.acquireLatestImage()
                    ?: return@setOnImageAvailableListener
                try {
                    val plane = image.planes.singleOrNull()
                        ?: throw IllegalStateException(
                            "JPEG camera image must contain one plane"
                        )
                    val buffer = plane.buffer
                    val jpeg = ByteArray(buffer.remaining())
                    buffer.get(jpeg)
                    val bitmap = BitmapFactory.decodeByteArray(
                        jpeg,
                        0,
                        jpeg.size,
                    ) ?: throw IllegalStateException(
                        "camera JPEG could not be decoded"
                    )
                    val prepared = try {
                        VN97BitmapVision.prepare(bitmap)
                    } finally {
                        bitmap.recycle()
                    }
                    finish(Result.success(prepared))
                } catch (exc: Throwable) {
                    finish(Result.failure(exc))
                } finally {
                    image.close()
                }
            },
            handler,
        )

        try {
            manager.openCamera(
                selection.cameraId,
                object : CameraDevice.StateCallback() {
                    override fun onOpened(opened: CameraDevice) {
                        if (finished.get()) {
                            opened.close()
                            return
                        }
                        device = opened
                        try {
                            opened.createCaptureSession(
                                listOf(reader.surface),
                                object :
                                    CameraCaptureSession.StateCallback() {
                                    override fun onConfigured(
                                        configured: CameraCaptureSession,
                                    ) {
                                        if (finished.get()) {
                                            configured.close()
                                            return
                                        }
                                        session = configured
                                        try {
                                            val request =
                                                opened.createCaptureRequest(
                                                    CameraDevice.TEMPLATE_STILL_CAPTURE
                                                ).apply {
                                                    addTarget(reader.surface)
                                                    set(
                                                        CaptureRequest.CONTROL_AF_MODE,
                                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE,
                                                    )
                                                    selection.sensorOrientation
                                                        ?.let { orientation ->
                                                            set(
                                                                CaptureRequest.JPEG_ORIENTATION,
                                                                orientation,
                                                            )
                                                        }
                                                }.build()
                                            configured.capture(
                                                request,
                                                object :
                                                    CameraCaptureSession.CaptureCallback() {},
                                                handler,
                                            )
                                        } catch (exc: Throwable) {
                                            finish(Result.failure(exc))
                                        }
                                    }

                                    override fun onConfigureFailed(
                                        configured: CameraCaptureSession,
                                    ) {
                                        finish(
                                            Result.failure(
                                                IllegalStateException(
                                                    "camera capture session configuration failed"
                                                )
                                            )
                                        )
                                    }
                                },
                                handler,
                            )
                        } catch (exc: Throwable) {
                            finish(Result.failure(exc))
                        }
                    }

                    override fun onDisconnected(
                        disconnected: CameraDevice,
                    ) {
                        finish(
                            Result.failure(
                                IllegalStateException(
                                    "camera disconnected"
                                )
                            )
                        )
                    }

                    override fun onError(
                        camera: CameraDevice,
                        error: Int,
                    ) {
                        finish(
                            Result.failure(
                                IllegalStateException(
                                    "camera error: $error"
                                )
                            )
                        )
                    }
                },
                handler,
            )
        } catch (exc: Throwable) {
            finish(Result.failure(exc))
        }
    }

    private data class CameraSelection(
        val cameraId: String,
        val size: Size,
        val sensorOrientation: Int?,
    )

    private fun selectCamera(
        manager: CameraManager,
    ): CameraSelection {
        val cameraIds = manager.cameraIdList
        require(cameraIds.isNotEmpty()) {
            "device exposes no cameras"
        }

        val selectedId = cameraIds.firstOrNull { id ->
            manager.getCameraCharacteristics(id)
                .get(CameraCharacteristics.LENS_FACING) ==
                CameraCharacteristics.LENS_FACING_BACK
        } ?: cameraIds.first()

        val characteristics =
            manager.getCameraCharacteristics(selectedId)
        val map = checkNotNull(
            characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            )
        ) {
            "camera stream configuration map unavailable"
        }
        val sizes = map.getOutputSizes(ImageFormat.JPEG)
            ?.filter { it.width > 0 && it.height > 0 }
            .orEmpty()
        require(sizes.isNotEmpty()) {
            "camera exposes no JPEG capture size"
        }
        val preferred = sizes
            .filter { it.width >= 224 && it.height >= 224 }
            .minByOrNull { it.width.toLong() * it.height.toLong() }
            ?: sizes.minByOrNull {
                it.width.toLong() * it.height.toLong()
            }
            ?: throw IllegalStateException(
                "camera capture size selection failed"
            )

        return CameraSelection(
            cameraId = selectedId,
            size = preferred,
            sensorOrientation =
                characteristics.get(
                    CameraCharacteristics.SENSOR_ORIENTATION
                ),
        )
    }
}
