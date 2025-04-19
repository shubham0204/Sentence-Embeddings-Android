package com.projects.shubham0204.demo

data class ModelConfig(
    val modelName: String,
    val modelAssetsFilepath: String,
    val tokenizerAssetsFilepath: String,
    val useTokenTypeIds: Boolean,
    val outputTensorName: String,
    val normalizeEmbeddings: Boolean,
)

enum class Model {
    ALL_MINILM_L6_V2,
    BGE_SMALL_EN_V1_5,
    SNOWFLAKE_ARCTIC_EMBED_S,
}

fun getModelConfig(model: Model): ModelConfig =
    when (model) {
        Model.ALL_MINILM_L6_V2 ->
            ModelConfig(
                modelName = "all-minilm-l6-v2",
                modelAssetsFilepath = "all-minilm-l6-v2/model.onnx",
                tokenizerAssetsFilepath = "all-minilm-l6-v2/tokenizer.json",
                useTokenTypeIds = true,
                outputTensorName = "last_hidden_state",
                normalizeEmbeddings = true,
            )

        Model.BGE_SMALL_EN_V1_5 ->
            ModelConfig(
                modelName = "bge-small-en-v1.5",
                modelAssetsFilepath = "bge-small-en-v1.5/model.onnx",
                tokenizerAssetsFilepath = "bge-small-en-v1.5/tokenizer.json",
                useTokenTypeIds = true,
                outputTensorName = "last_hidden_state",
                normalizeEmbeddings = true,
            )

        Model.SNOWFLAKE_ARCTIC_EMBED_S ->
            ModelConfig(
                modelName = "snowflake-arctic-embed-s",
                modelAssetsFilepath = "snowflake-arctic-embed-s/model.onnx",
                tokenizerAssetsFilepath = "snowflake-arctic-embed-s/tokenizer.json",
                useTokenTypeIds = true,
                outputTensorName = "last_hidden_state",
                normalizeEmbeddings = true,
            )
    }
