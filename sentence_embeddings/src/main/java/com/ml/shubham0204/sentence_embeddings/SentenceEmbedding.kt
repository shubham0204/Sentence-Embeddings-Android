package com.ml.shubham0204.sentence_embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer

class SentenceEmbedding(
    private val context: Context
) {

    private lateinit var hfTokenizer: HFTokenizer
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession

    suspend fun init(
        modelAssetsPath: String,
        tokenizerAssetsPath: String
    ) =
        withContext(Dispatchers.IO) {
            hfTokenizer = HFTokenizer(tokenizerAssetsPath)
            ortEnvironment = OrtEnvironment.getEnvironment()
            ortSession =
                ortEnvironment.createSession(
                    context.assets.open(modelAssetsPath).readBytes()
                )
        }

    fun encode(
        sentence: String
    ): FloatArray {
        val result = hfTokenizer.tokenize(sentence)
        val idsTensor =
            OnnxTensor.createTensor(
                ortEnvironment,
                LongBuffer.wrap(result.ids),
                longArrayOf(1, result.ids.size.toLong()),
            )
        val attentionMaskTensor =
            OnnxTensor.createTensor(
                ortEnvironment,
                LongBuffer.wrap(result.attentionMask),
                longArrayOf(1, result.attentionMask.size.toLong()),
            )
        val outputs =
            ortSession.run(mapOf("input_ids" to idsTensor, "attention_mask" to attentionMaskTensor))
        val embeddingTensor = outputs.get("sentence_embedding").get() as OnnxTensor
        return embeddingTensor.floatBuffer.array()
    }
}
