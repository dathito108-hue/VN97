package ai.vn97.app

import ai.vn97.avatar.AudioLevelMeter
import ai.vn97.runtime.NativeAudioModality
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import java.util.concurrent.atomic.AtomicBoolean

data class VN97VoiceUtterance(
    val pcm16: ShortArray,
    val sampleRateHz: Int = NativeAudioModality.SAMPLE_RATE_HZ,
) {
    init {
        require(pcm16.isNotEmpty())
        require(pcm16.size <= NativeAudioModality.MAX_UTTERANCE_SAMPLES)
        require(sampleRateHz == NativeAudioModality.SAMPLE_RATE_HZ)
    }

    val durationMillis: Long
        get() = pcm16.size.toLong() * 1000L / sampleRateHz
}

internal class VN97VoiceCapture(
    private val context: Context,
    private val onLevel: (Float) -> Unit,
    private val onUtterance: (VN97VoiceUtterance) -> Unit,
    private val onFailure: (String) -> Unit,
) : AutoCloseable {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val recording = AtomicBoolean(false)
    @Volatile private var audioRecord: AudioRecord? = null
    @Volatile private var worker: Thread? = null

    val isRecording: Boolean
        get() = recording.get()

    fun hasPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        if (!hasPermission()) return false
        if (!recording.compareAndSet(false, true)) return true

        val minBuffer = AudioRecord.getMinBufferSize(
            NativeAudioModality.SAMPLE_RATE_HZ,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuffer <= 0) {
            recording.set(false)
            fail("Microphone buffer configuration is unavailable.")
            return false
        }

        val bufferBytes = maxOf(
            minBuffer,
            NativeAudioModality.FRAME_SIZE * Short.SIZE_BYTES * 4,
        )
        val recorder = try {
            AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(NativeAudioModality.SAMPLE_RATE_HZ)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build()
                )
                .setBufferSizeInBytes(bufferBytes)
                .build()
        } catch (exc: Throwable) {
            recording.set(false)
            fail("Microphone open failed: " + exc::class.java.simpleName)
            return false
        }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recording.set(false)
            recorder.release()
            fail("Microphone could not initialize at 16 kHz PCM16.")
            return false
        }

        audioRecord = recorder
        val thread = Thread(
            { captureLoop(recorder, bufferBytes / Short.SIZE_BYTES) },
            "vn97-voice-capture",
        )
        worker = thread
        thread.start()
        return true
    }

    fun stop() {
        recording.set(false)
    }

    @SuppressLint("MissingPermission")
    private fun captureLoop(recorder: AudioRecord, bufferSamples: Int) {
        val chunk = ShortArray(bufferSamples.coerceAtLeast(NativeAudioModality.FRAME_SIZE))
        val utterance = ShortArray(NativeAudioModality.MAX_UTTERANCE_SAMPLES)
        var written = 0
        var failure: String? = null

        try {
            recorder.startRecording()
            if (recorder.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                failure = "Microphone did not enter recording state."
                return
            }

            while (recording.get() && written < utterance.size) {
                val remaining = utterance.size - written
                val requested = minOf(chunk.size, remaining)
                val read = recorder.read(
                    chunk,
                    0,
                    requested,
                    AudioRecord.READ_BLOCKING,
                )
                when {
                    read > 0 -> {
                        chunk.copyInto(
                            destination = utterance,
                            destinationOffset = written,
                            startIndex = 0,
                            endIndex = read,
                        )
                        written += read
                        val level = AudioLevelMeter.rmsPcm16(chunk, 0, read)
                        mainHandler.post {
                            if (recording.get()) onLevel(level)
                        }
                    }
                    read == 0 -> Unit
                    else -> {
                        failure = "Microphone read failed with code " + read
                        break
                    }
                }
            }
        } catch (exc: Throwable) {
            failure = "Microphone capture failed: " + exc::class.java.simpleName
        } finally {
            recording.set(false)
            try {
                if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                    recorder.stop()
                }
            } catch (_: IllegalStateException) {
                // Recorder may already be stopped by the platform.
            }
            recorder.release()
            audioRecord = null
            worker = null
            mainHandler.post { onLevel(0f) }

            val message = failure
            if (message != null) {
                fail(message)
            } else if (written >= NativeAudioModality.FRAME_SIZE) {
                val samples = utterance.copyOf(written)
                mainHandler.post {
                    onUtterance(VN97VoiceUtterance(samples))
                }
            } else if (written > 0) {
                fail("Voice capture was shorter than one 20 ms VN97 audio frame.")
            }
        }
    }

    private fun fail(message: String) {
        mainHandler.post { onFailure(message) }
    }

    override fun close() {
        recording.set(false)
        val recorder = audioRecord
        if (recorder != null) {
            try {
                recorder.stop()
            } catch (_: IllegalStateException) {
                // Capture loop owns normal shutdown.
            }
        }
    }
}
