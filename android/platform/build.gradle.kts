plugins {
    id("com.android.library")
}

android {
    namespace = "ai.vn97.platform"
    compileSdk = 37

    defaultConfig {
        minSdk = 26
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
    testImplementation("junit:junit:4.13.2")
}

tasks.withType<org.gradle.api.tasks.testing.Test>().configureEach {
    testLogging { events("passed", "failed", "skipped") }
}
