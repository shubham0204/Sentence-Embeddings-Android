// The rust-android-gradle plugin has to be declared before the Android/Kotlin plugins
// see: https://github.com/mozilla/rust-android-gradle/issues/147#issuecomment-2134688017
plugins {
    id("org.mozilla.rust-android-gradle.rust-android") version "0.9.6"
    alias(libs.plugins.android.library)
    alias(libs.plugins.jetbrains.kotlin.android)
}

// Used in GitHub CI to pass the path of the installed Android NDK
val envAndroidNDKPath = System.getenv("ANDROID_NDK_HOME")

android {
    namespace = "com.ml.shubham0204.model2vec"
    compileSdk = 35

    // Declare the ndkVersion to avoid 'NDK not installed' errors from rust-android-plugin
    // see: https://github.com/mozilla/rust-android-gradle/issues/29#issuecomment-593501017
    ndkVersion = "27.2.12479018" // Android NDK r27c
    envAndroidNDKPath?.let {
        ndkPath = it
    }

    defaultConfig {
        minSdk = 24

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
    kotlinOptions {
        jvmTarget = "17"
    }
}

cargo {
    module = "../rs-model2vec"
    libname = "model2vec"
    prebuiltToolchains = true
    targets = listOf("arm", "arm64", "x86", "x86_64")
    profile = "release"
    verbose = true
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
