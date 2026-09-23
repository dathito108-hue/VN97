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

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        warningsAsErrors = true
    }
}

dependencies {
    implementation(project(":runtime"))
    implementation(project(":platform"))
    implementation(project(":avatar"))
}
