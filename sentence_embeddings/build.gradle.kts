// The rust-android-gradle plugin has to be declared before the Android/Kotlin plugins
// see: https://github.com/mozilla/rust-android-gradle/issues/147#issuecomment-2134688017
plugins {
    id("org.mozilla.rust-android-gradle.rust-android") version "0.9.6"
    alias(libs.plugins.android.library)
    alias(libs.plugins.jetbrains.kotlin.android)
    id("maven-publish")
}

android {
    namespace = "com.ml.shubham0204.sentence_embeddings"
    compileSdk = 34

    // Declare the ndkVersion to avoid 'NDK not installed' errors from rust-android-plugin
    // see: https://github.com/mozilla/rust-android-gradle/issues/29#issuecomment-593501017
    ndkVersion = "27.0.12077973"
    defaultConfig {
        minSdk = 26

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    androidResources {
        noCompress += "onnx"
    }
}

cargo {
    module = "../rs-hf-tokenizer"
    libname = "hftokenizer"
    prebuiltToolchains = true
    targets = listOf("arm", "arm64", "x86", "x86_64")
    verbose = true
    profile = "release"
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")
}

afterEvaluate {
    publishing {
        publications {
            create("release", MavenPublication::class.java) {
                from(components["release"])
                groupId = "com.github.shubham0204"
                artifactId = "sentence_embeddings"
                version = "0.1"
            }
        }
    }
}
