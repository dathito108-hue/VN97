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
        val installer = VN97G06BundledRuntime(VN97Application(privateRoot, manager)) { staging ->
            check(!File(staging, "binding.vn97m2g09.json").readText().contains("reject"))
        }
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
        println("G06 bundled installer: fresh install, repeat, update and failure preservation PASS")
    } finally {
        check(temp.deleteRecursively())
    }
}
