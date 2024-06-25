package com.ml.shubham0204.sentence_embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import java.util.EnumSet

class SentenceEmbedding {

    private lateinit var hfTokenizer: HFTokenizer
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession

    suspend fun init(
        modelBytes: ByteArray,
        tokenizerBytes: ByteArray,
        useFP16: Boolean = false,
        useXNNPack: Boolean = false
    ) = withContext(Dispatchers.IO) {
        hfTokenizer = HFTokenizer(tokenizerBytes)
        ortEnvironment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply{
            if (useFP16) {
                addNnapi(EnumSet.of(NNAPIFlags.USE_FP16, NNAPIFlags.CPU_DISABLED))
            }
            if (useXNNPack) {
                addXnnpack(mapOf(
                    "intra_op_num_threads" to "2"
                ))
            }
        }
        ortSession = ortEnvironment.createSession(modelBytes,options)
    }

    suspend fun encode(
        sentence: String
    ): FloatArray = withContext(Dispatchers.IO) {
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
        return@withContext embeddingTensor.floatBuffer.array()
    }
}
