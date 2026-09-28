package ai.vn97.app

import android.content.res.AssetManager
import java.io.File
import java.nio.file.Files

private fun writeFixture(root: File, relative: String, text: String) {
    val file = File(root, "vn97-g06/$relative")
    file.parentFile.mkdirs()
    file.writeText(text)
}

fun main() {
    val temp = Files.createTempDirectory("vn97-g06-bundle-test").toFile()
    try {
        val assets = File(temp, "assets").apply { mkdirs() }
        val privateRoot = File(temp, "private").apply { mkdirs() }
        val manager = AssetManager(assets)
        var validations = 0
        fun newInstaller() = VN97G06BundledRuntime(VN97Application(privateRoot, manager)) { staging ->
            validations++
            check(!File(staging, "binding.vn97m2g09.json").readText().contains("reject"))
            // Stand-in for the production validator's graph/tokenizer hash checks.
            listOf("runtime/recurrent-8.onnx", "tokenizer/vocab.json", "tokenizer/merges.txt").forEach {
                check(File(staging, it).readText() == "fixture:$it:v1")
            }
        }
        var installer = newInstaller()
        check(!installer.installIfPresent(required = false))
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)

        val files = listOf(
            "binding.vn97m2g09.json", "tuning.vn97m2g07.json",
            "runtime/runtime.vn97m2g06.json", "runtime/recurrent-8.onnx",
            "promotion.vn97m2g10.json", "tokenizer/tokenizer.vn97m2g08.json",
            "tokenizer/vocab.json", "tokenizer/merges.txt",
        )
        files.forEach { writeFixture(assets, it, "fixture:$it:v1") }
        // Both the staging root and nested parents already exist for sibling files.
        check(installer.installIfPresent(required = true))
        val target = File(privateRoot, "vn97-g06")
        files.forEach { check(File(target, it).readText() == "fixture:$it:v1") }
        val identity = File(target, ".apk-assets.sha256").readText()
        check(identity.trim().matches(Regex("[0-9a-f]{64}")))
        check(!installer.installIfPresent(required = true))
        check(File(target, ".apk-assets.sha256").readText() == identity)
        check(!File(privateRoot, ".vn97-g06.staging").exists())

        // Reopening in the same process must not recopy immutable APK bytes.
        manager.failOpenPath = "vn97-g06/runtime/recurrent-8.onnx"
        check(!installer.installIfPresent(required = true))
        manager.failOpenPath = null

        // A new process verifies installed payloads without opening/copying the
        // APK graph. Stale temporary directories are removed only after success.
        val stale = File(privateRoot, ".vn97-g06.staging/partial")
        stale.parentFile.mkdirs()
        stale.writeText("partial")
        val oldBackup = File(privateRoot, ".vn97-g06.backup/old")
        oldBackup.parentFile.mkdirs()
        oldBackup.writeText("old")
        val before = validations
        manager.failOpenPath = "vn97-g06/runtime/recurrent-8.onnx"
        check(!newInstaller().installIfPresent(required = true))
        check(validations == before + 1)
        check(!stale.parentFile.exists() && !oldBackup.parentFile.exists())
        manager.failOpenPath = null

        // A self-consistent local installation is not enough: even an activation
        // receipt change in the trusted APK must take the replacement path.
        writeFixture(assets, "promotion.vn97m2g10.json", "fixture:promotion:v2")
        manager.failOpenPath = "vn97-g06/runtime/recurrent-8.onnx"
        check(runCatching { newInstaller().installIfPresent(required = true) }.isFailure)
        check(File(target, "promotion.vn97m2g10.json").readText() == "fixture:promotion.vn97m2g10.json:v1")
        writeFixture(assets, "promotion.vn97m2g10.json", "fixture:promotion.vn97m2g10.json:v1")
        manager.failOpenPath = null

        // Matching descriptors and a stale identity marker cannot hide damage.
        File(target, "runtime/recurrent-8.onnx").writeText("corrupt")
        check(newInstaller().installIfPresent(required = true))
        check(File(target, "runtime/recurrent-8.onnx").readText() == "fixture:runtime/recurrent-8.onnx:v1")
        File(target, "tokenizer/vocab.json").writeText("corrupt")
        check(newInstaller().installIfPresent(required = true))
        check(File(target, "tokenizer/vocab.json").readText() == "fixture:tokenizer/vocab.json:v1")

        // A new APK/process has a new installer; its assets may differ.
        installer = newInstaller()
        // Partial new staging data must never replace the working installation.
        writeFixture(assets, "binding.vn97m2g09.json", "fixture:v2")
        manager.failOpenPath = "vn97-g06/runtime/recurrent-8.onnx"
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)
        files.forEach { check(File(target, it).readText() == "fixture:$it:v1") }
        check(File(target, ".apk-assets.sha256").readText() == identity)
        check(!File(privateRoot, ".vn97-g06.staging").exists())
        manager.failOpenPath = null

        check(installer.installIfPresent(required = true))
        check(File(target, "binding.vn97m2g09.json").readText() == "fixture:v2")
        val secondIdentity = File(target, ".apk-assets.sha256").readText()
        check(secondIdentity != identity)
        check(!File(privateRoot, ".vn97-g06.backup").exists())

        installer = newInstaller()
        // Semantic rejection happens before replacing the validated installation.
        writeFixture(assets, "binding.vn97m2g09.json", "reject")
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)
        check(File(target, ".apk-assets.sha256").readText() == secondIdentity)
        writeFixture(assets, "binding.vn97m2g09.json", "fixture:v2")

        // Missing required graphs fail before the old target is moved.
        File(assets, "vn97-g06/runtime/recurrent-8.onnx").delete()
        check(runCatching { installer.installIfPresent(required = true) }.isFailure)
        check(File(target, ".apk-assets.sha256").readText() == secondIdentity)
        check(File(target, "runtime/recurrent-8.onnx").readText() == "fixture:runtime/recurrent-8.onnx:v1")
        check(!File(privateRoot, ".vn97-g06.staging").exists())
        // Simulate process death in the directory-swap gap, with no APK assets
        // available on restart. Recovery must not depend on another 5GB copy.
        val backup = File(privateRoot, ".vn97-g06.backup")
        check(target.renameTo(backup))
        check(File(assets, "vn97-g06").deleteRecursively())
        val partial = File(privateRoot, ".vn97-g06.staging/partial")
        partial.parentFile.mkdirs()
        partial.writeText("incomplete")
        installer = newInstaller()
        check(!installer.installIfPresent(required = false))
        check(File(target, ".apk-assets.sha256").readText() == secondIdentity)
        check(!backup.exists())
        check(!File(target, "partial").exists())

        // A completed swap must not be rolled back to a leftover backup.
        backup.mkdirs()
        File(backup, "old").writeText("old")
        check(!newInstaller().installIfPresent(required = false))
        check(File(target, ".apk-assets.sha256").readText() == secondIdentity)
        check(backup.deleteRecursively())

        // Do not recover a semantically invalid backup or destroy its evidence.
        check(target.renameTo(backup))
        File(backup, "binding.vn97m2g09.json").writeText("reject")
        check(runCatching { newInstaller().installIfPresent(required = false) }.isFailure)
        check(!target.exists())
        check(backup.isDirectory)
        println("G06 bundled installer: authenticated restart without copying, corruption repair, update and recovery PASS")
    } finally {
        check(temp.deleteRecursively())
    }
}
