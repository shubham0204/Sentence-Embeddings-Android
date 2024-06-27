# Sentence Embeddings in Android

> An Android library that provides a port to sentence-transformers, which are used to generate sentence embeddings (fixed-size vectors for text/sentences)

[![](https://jitpack.io/v/shubham0204/Sentence-Embeddings-Android.svg)](https://jitpack.io/#shubham0204/Sentence-Embeddings-Android)

![App Demo](resources/app_demo.gif)

## Setup

### 1. Add the Jitpack repository to `settings.gradle.kts`

The library is hosted with [Jitpack](https://jitpack.io/). Add the `jitpack.io` repository in `settings.gradle.kts` for Gradle to search Jitpack packages,

```kotlin
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven{ url = uri("https://jitpack.io") }
    }
}
```

or with Groovy build scripts,

```groovy
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url "https://jitpack.io" }
    }
}
```

### 2. Add the dependency to `build.gradle.kts`

Add the `Sentence-Embeddings-Android` dependency to `build.gradle.kts`,

```kotlin
dependencies {
    // ...
    implementation("com.github.shubham0204:Sentence-Embeddings-Android:0.0.3")
    // ...
}
```

Sync the Gradle scripts and rebuild the project.

### 3. (Optional) Download the ONNX model and `tokenizer.json` for `all-MiniLM-L6-V2`

> [!NOTE]
> You may download the model and the tokenizer at runtime, as the library only expects raw-bytes of these files. If you wish to include them in the app's package, then proceed with this step

The ONNX model and the tokenizer can be downloaded from the [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) repository,

- Download [`model.onnx`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/onnx/model.onnx)
- Download [`tokenizer.json`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/tokenizer.json)

Place `model.onnx` and `tokenizer.json` in the `assets` folder of the application. 

## Usage

### API

The library provides a `SentenceEmbedding` class with `init` and `encode` suspend functions that initialize the model and generate the sentence embedding respectively. 

The `init` function takes two mandatory arguments, `modelBytes` and `tokenizerBytes`.

```kotlin
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding

val sentenceEmbedding = SentenceEmbedding()
val modelBytes: ByteArray = context.assets.open("all-MiniLM-L6-V2.onnx").use{ it.readBytes() }
val tokenizerBytes: ByteArray = context.assets.open("tokenizer.json").use{ it.readBytes() }
CoroutineScope(Dispatchers.IO).launch {
    sentenceEmbedding.init(
        modelBytes,
        tokenizerBytes
    )
}
```

Once the `init` functions completes its execution, we can call the `encode` function to transform the given `sentence` to an embedding,

```kotlin
CoroutineScope(Dispatchers.IO).launch {
    val embedding: FloatArray = sentenceEmbedding.encode( "Delhi has a population 32 million" )
    println( "Embedding: $embedding" )
    println( "Embedding size: ${embedding.size}")
}
```

### Compute Cosine Similarity

The embeddings are vectors whose relative similarity can be computed by measuring the cosine of the angle between the vectors, also termed as *cosine similarity*,

> [!TIP]
> Here's an excellent [blog](https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db) to under cosine similarity

```kotlin
private fun cosineDistance(
    x1: FloatArray,
    x2: FloatArray
): Float {
    var mag1 = 0.0f
    var mag2 = 0.0f
    var product = 0.0f
    for (i in x1.indices) {
        mag1 += x1[i].pow(2)
        mag2 += x2[i].pow(2)
        product += x1[i] * x2[i]
    }
    mag1 = sqrt(mag1)
    mag2 = sqrt(mag2)
    return product / (mag1 * mag2)
}

CoroutineScope(Dispatchers.IO).launch {
    val e1: FloatArray = sentenceEmbedding.encode( "Delhi has a population 32 million" )
    val e2: FloatArray = sentenceEmbedding.encode( "What is the population of Delhi?" )
    val e3: FloatArray = sentenceEmbedding.encode( "Cities with a population greater than 4 million are termed as metro cities" )
    
    val d12 = cosineDistance( e1 , e2 )
    val d13 = cosineDistance( e1 , e3 )
    println( "Similarity between e1 and e2: $d12" )
    println( "Similarity between e1 and e3: $d13" )
}
```