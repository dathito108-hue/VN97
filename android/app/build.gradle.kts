plugins {
    id("com.android.application")
}

android {
    namespace = "ai.vn97.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "ai.vn97.app"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
    }

    buildFeatures {
        buildConfig = true
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
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        warningsAsErrors = true
    }
}

val turnkeyBootstrapDir =
    layout.projectDirectory.dir("src/main/assets/vn97-bootstrap")

val turnkeyBootstrapFiles = listOf(
    "model.vn97cap1",
    "model.vn97sig1",
    "publisher.ed25519",
)

val verifyTurnkeyBootstrap by tasks.registering {
    group = "verification"
    description =
        "Require a complete non-empty signed VN97 bootstrap for release APKs."

    inputs.dir(turnkeyBootstrapDir)

    doLast {
        val root = turnkeyBootstrapDir.asFile
        val missing = turnkeyBootstrapFiles.filter { name ->
            val file = root.resolve(name)
            !file.isFile || file.length() <= 0L
        }
        check(missing.isEmpty()) {
            "VN97 turnkey release requires bundled bootstrap assets: " +
                missing.joinToString(", ")
        }
    }
}

tasks.matching { it.name == "preReleaseBuild" }.configureEach {
    dependsOn(verifyTurnkeyBootstrap)
}

dependencies {
    implementation(project(":runtime"))
    implementation(project(":platform"))
    implementation(project(":avatar"))
}
