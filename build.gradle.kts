// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.jetbrains.kotlin.android) apply false
    alias(libs.plugins.android.library) apply false
    id("org.mozilla.rust-android-gradle.rust-android") version "0.9.6" apply false
    alias(libs.plugins.kotlin.compose) apply false
}