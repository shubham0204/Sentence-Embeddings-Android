- Move the Rust source code from the `libs` branch to the `main` branch: We now use
  the [rust-android-plugin](https://github.com/mozilla/rust-android-gradle/issues/29) to initiate
  `cargo build` from Gradle

- Removed Git LFS: The ONNX models present in `app/src/main/assets` have been removed from the
  repository. Instead, `app/build.gradle.kts` downloads the models and tokenizer configs from
  HuggingFace using `download_model.sh` shell script.

- Add [Model2Vec](https://huggingface.co/blog/Pringled/model2vec): Model2Vec provides static
  sentence-embeddings through a fast-lookup

- Remove Jitpack: A GitHub CI script now builds AARs for `model2vec` and `sentence_embeddings`
  Gradle modules that can be included in other projects