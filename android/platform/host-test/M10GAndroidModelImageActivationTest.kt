package ai.vn97.platform

import android.content.Context
import java.nio.file.Files

fun main() {
    val file = Files.createTempFile("m10g-android-", ".vn97mi1").toFile()
    file.writeBytes(ByteArray(32) { 1 })
    AndroidVN97ModelImageCandidateValidator.validate(
        file,
        ByteArray(32) { 2 },
    )

    val root = Files.createTempDirectory("m10g-app-").toFile()
    val provisioner = AndroidVN97CapabilityProvisioner(Context(root))
    check(
        provisioner.compatibilityProfile.profileId ==
            "vn97.android.model-image.v1"
    )
    check(provisioner.stageRoot.isDirectory)
    provisioner.createStager()

    println("M10G_ANDROID_VALIDATOR_SYNTAX_PASS")
}
