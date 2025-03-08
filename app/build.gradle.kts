plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.projects.shubham0204.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.projects.shubham0204.demo"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        vectorDrawables {
            useSupportLibrary = true
        }
    }

    signingConfigs {
        create("release") {
            storeFile = file("../keystore.jks")
            storePassword = System.getenv("RELEASE_KEYSTORE_PASSWORD")
            keyAlias = System.getenv("RELEASE_KEYSTORE_ALIAS")
            keyPassword = System.getenv("RELEASE_KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

// Task to download models if assets directory is empty
tasks.register("downloadModelsIfNeeded") {
    description = "Downloads ONNX models if the assets directory is empty"
    doLast {
        val assetsDir = File("${project.projectDir}/src/main/assets")
        val isWindowsOS = System.getProperty("os.name").lowercase().contains("windows")
        val scriptFile = if (isWindowsOS) "../download_models.bat" else "../download_models.sh"
        logger.lifecycle("Checking if ONNX models need to be downloaded...")
        val requiredModelDirs =
            listOf(
                "all-minilm-l6-v2",
                "bge-small-en-v1.5",
                "snowflake-arctic-embed-s",
            )
        val needsDownload =
            !assetsDir.exists() ||
                !assetsDir.isDirectory ||
                assetsDir.listFiles()?.isEmpty() ?: true ||
                requiredModelDirs.any { dir ->
                    val modelDir = File(assetsDir, dir)
                    !modelDir.exists() ||
                        !File(modelDir, "model.onnx").exists() ||
                        !File(modelDir, "tokenizer.json").exists()
                }

        if (needsDownload) {
            logger.lifecycle("Assets directory is empty or missing required models. Downloading models...")
            if (!isWindowsOS) {
                exec {
                    workingDir = project.rootDir
                    commandLine("chmod", "+x", scriptFile.removePrefix("../"))
                }
            }
            exec {
                workingDir = project.rootDir
                if (isWindowsOS) {
                    commandLine("cmd", "/c", scriptFile.removePrefix("../"))
                } else {
                    commandLine("sh", "-c", "./${scriptFile.removePrefix("../")}")
                }
            }
            logger.lifecycle("Model download completed.")
        } else {
            logger.lifecycle("Models already exist in assets directory. Skipping download.")
        }
    }
}

tasks.named("preBuild") {
    dependsOn("downloadModelsIfNeeded")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)

    implementation(project(":sentence_embeddings"))
}
