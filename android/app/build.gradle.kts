import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.security.MessageDigest
import java.util.Base64

plugins {
    id("com.android.application")
}

val vn97VersionCode = 190100
val vn97VersionName = "1.0.0-rc1"

val releaseKeystorePath =
    providers.environmentVariable(
        "VN97_RELEASE_KEYSTORE"
    ).orNull
val releaseStorePassword =
    providers.environmentVariable(
        "VN97_RELEASE_STORE_PASSWORD"
    ).orNull
val releaseKeyAlias =
    providers.environmentVariable(
        "VN97_RELEASE_KEY_ALIAS"
    ).orNull
val releaseKeyPassword =
    providers.environmentVariable(
        "VN97_RELEASE_KEY_PASSWORD"
    ).orNull

val releaseSigningValues =
    listOf(
        releaseKeystorePath,
        releaseStorePassword,
        releaseKeyAlias,
        releaseKeyPassword,
    )

android {
    namespace = "ai.vn97.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "ai.vn97.app"
        minSdk = 26
        targetSdk = 37
        versionCode = vn97VersionCode
        versionName = vn97VersionName
    }

    buildFeatures {
        buildConfig = true
    }

    signingConfigs {
        create("release") {
            if (
                releaseSigningValues.all {
                    !it.isNullOrBlank()
                }
            ) {
                storeFile =
                    file(
                        checkNotNull(
                            releaseKeystorePath
                        )
                    )
                storePassword =
                    releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword =
                    releaseKeyPassword
            }
        }
    }

    buildTypes {
        debug {
            buildConfigField(
                "boolean",
                "VN97_TURNKEY_REQUIRED",
                "false",
            )
        }
        release {
            buildConfigField(
                "boolean",
                "VN97_TURNKEY_REQUIRED",
                "true",
            )
            signingConfig =
                signingConfigs
                    .getByName("release")
        }
    }

    sourceSets {
        getByName("release").assets.srcDir(
            layout.buildDirectory
                .dir(
                    "generated/vn97-release/assets"
                )
                .get()
                .asFile
        )
    }

    compileOptions {
        sourceCompatibility =
            JavaVersion.VERSION_17
        targetCompatibility =
            JavaVersion.VERSION_17
    }

    lint {
        warningsAsErrors = true
    }
}

val turnkeyBootstrapDir =
    layout.projectDirectory.dir(
        "src/main/assets/vn97-bootstrap"
    )

data class TurnkeyAssetRule(
    val name: String,
    val minBytes: Long,
    val maxBytes: Long,
)

val turnkeyBootstrapRules =
    listOf(
        TurnkeyAssetRule(
            name = "model.vn97cap1",
            minBytes = 96L,
            maxBytes =
                512L * 1024L * 1024L,
        ),
        TurnkeyAssetRule(
            name = "model.vn97sig1",
            minBytes = 1L,
            maxBytes = 16L * 1024L,
        ),
        TurnkeyAssetRule(
            name = "publisher.ed25519",
            minBytes = 32L,
            maxBytes = 32L,
        ),
    )

val verifyTurnkeyBootstrap by tasks.registering {
    group = "verification"
    description =
        "Require a safe complete signed VN97 bootstrap for release APKs."

    inputs.dir(turnkeyBootstrapDir)

    doLast {
        val root =
            turnkeyBootstrapDir.asFile
        val rootPath = root.toPath()
        check(
            Files.isDirectory(
                rootPath,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(
                    rootPath
                )
        ) {
            "VN97 turnkey bootstrap root must be a real directory."
        }

        val allowedEntries =
            turnkeyBootstrapRules
                .map { it.name }
                .toSet() +
                "README.txt"
        val entries =
            root.listFiles()
                ?.toList()
                .orEmpty()
        val unexpected =
            entries.map { it.name }
                .filter {
                    it !in allowedEntries
                }
        check(unexpected.isEmpty()) {
            "VN97 turnkey bootstrap contains unexpected files: " +
                unexpected.joinToString(", ")
        }

        turnkeyBootstrapRules.forEach {
            rule ->
            val file =
                root.resolve(rule.name)
            val path = file.toPath()
            check(
                Files.isRegularFile(
                    path,
                    LinkOption.NOFOLLOW_LINKS,
                ) &&
                    !Files.isSymbolicLink(
                        path
                    )
            ) {
                "VN97 turnkey release requires a safe regular asset: " +
                    rule.name
            }
            val size = Files.size(path)
            check(
                size in
                    rule.minBytes..
                        rule.maxBytes
            ) {
                "VN97 turnkey asset size is outside bounds: " +
                    rule.name
            }
        }

        val packageFile =
            root.resolve(
                "model.vn97cap1"
            )
        val magic =
            packageFile.inputStream().use {
                input ->
                val bytes = ByteArray(8)
                val count =
                    input.read(bytes)
                check(count == bytes.size) {
                    "VN97CAP1 bootstrap header is truncated."
                }
                bytes.toString(
                    StandardCharsets.US_ASCII
                )
            }
        check(magic == "VN97CAP1") {
            "VN97 turnkey model asset is not VN97CAP1."
        }
    }
}

val verifyReleaseSigning by tasks.registering {
    group = "verification"
    description =
        "Require external non-repository signing material for VN97 release APKs."

    doLast {
        check(
            releaseSigningValues.all {
                !it.isNullOrBlank()
            }
        ) {
            "VN97 release signing requires VN97_RELEASE_KEYSTORE, " +
                "VN97_RELEASE_STORE_PASSWORD, VN97_RELEASE_KEY_ALIAS, " +
                "and VN97_RELEASE_KEY_PASSWORD."
        }

        val store =
            file(
                checkNotNull(
                    releaseKeystorePath
                )
            )
        val storePath =
            store.toPath()
                .toAbsolutePath()
                .normalize()
        check(
            Files.isRegularFile(
                storePath,
                LinkOption.NOFOLLOW_LINKS,
            ) &&
                !Files.isSymbolicLink(
                    storePath
                ) &&
                Files.size(storePath) > 0L
        ) {
            "VN97 release keystore must be a non-empty regular file."
        }

        val repositoryRoot =
            rootProject.projectDir
                .parentFile
                .toPath()
                .toAbsolutePath()
                .normalize()
        check(
            !storePath.startsWith(
                repositoryRoot
            )
        ) {
            "VN97 release keystore must stay outside the repository."
        }
    }
}

val verifyReleaseVersion by tasks.registering {
    group = "verification"
    description =
        "Verify the canonical M19 release application/version identity."

    doLast {
        check(vn97VersionCode >= 190100) {
            "VN97 production versionCode is below M19 release floor."
        }
        check(
            vn97VersionName.matches(
                Regex(
                    "^1\\.0\\.0(?:-rc[0-9]+)?$"
                )
            )
        ) {
            "VN97 production versionName is outside the M19 contract."
        }
    }
}

val releaseManifestFile =
    layout.buildDirectory.file(
        "generated/vn97-release/assets/" +
            "vn97-release/release.vn97rel1"
    )

fun sha256(file: java.io.File): String {
    val digest =
        MessageDigest.getInstance(
            "SHA-256"
        )
    file.inputStream().buffered().use {
        input ->
        val buffer = ByteArray(64 * 1024)
        while (true) {
            val count =
                input.read(buffer)
            if (count < 0) break
            check(count > 0) {
                "VN97 release digest read made no progress."
            }
            digest.update(
                buffer,
                0,
                count,
            )
        }
    }
    return digest.digest()
        .joinToString("") {
            "%02x".format(
                it.toInt() and 0xff
            )
        }
}

val writeTurnkeyReleaseManifest by tasks.registering {
    group = "build"
    description =
        "Generate VN97REL1 binding release version to exact bootstrap identities."
    dependsOn(
        verifyTurnkeyBootstrap,
        verifyReleaseVersion,
    )

    inputs.dir(turnkeyBootstrapDir)
    outputs.file(releaseManifestFile)

    doLast {
        val root =
            turnkeyBootstrapDir.asFile
        val model =
            root.resolve(
                "model.vn97cap1"
            )
        val signature =
            root.resolve(
                "model.vn97sig1"
            )
        val publisher =
            root.resolve(
                "publisher.ed25519"
            )
        val output =
            releaseManifestFile.get()
                .asFile
        output.parentFile.mkdirs()

        val encodedVersion =
            Base64.getEncoder()
                .encodeToString(
                    vn97VersionName
                        .toByteArray(
                            StandardCharsets.UTF_8
                        )
                )
        val content =
            buildString {
                append("VN97REL1\n")
                append(
                    "application_id=ai.vn97.app\n"
                )
                append(
                    "version_code=" +
                        vn97VersionCode +
                        "\n"
                )
                append(
                    "version_name_b64=" +
                        encodedVersion +
                        "\n"
                )
                append(
                    "model_bytes=" +
                        model.length() +
                        "\n"
                )
                append(
                    "model_sha256=" +
                        sha256(model) +
                        "\n"
                )
                append(
                    "signature_bytes=" +
                        signature.length() +
                        "\n"
                )
                append(
                    "signature_sha256=" +
                        sha256(signature) +
                        "\n"
                )
                append(
                    "publisher_bytes=" +
                        publisher.length() +
                        "\n"
                )
                append(
                    "publisher_sha256=" +
                        sha256(publisher) +
                        "\n"
                )
            }
        output.writeText(
            content,
            StandardCharsets.UTF_8,
        )
    }
}

tasks.matching {
    it.name == "mergeReleaseAssets"
}.configureEach {
    dependsOn(
        writeTurnkeyReleaseManifest
    )
}

tasks.matching {
    it.name == "preReleaseBuild"
}.configureEach {
    dependsOn(
        verifyTurnkeyBootstrap,
        verifyReleaseSigning,
        verifyReleaseVersion,
    )
}

dependencies {
    implementation(project(":runtime"))
    implementation(project(":platform"))
    implementation(project(":avatar"))
}
