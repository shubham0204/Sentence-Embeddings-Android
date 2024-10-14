package com.ml.shubham0204.sentence_embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.max

class SentenceEmbedding {

    private lateinit var hfTokenizer: HFTokenizer
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var useTokenTypeIds: Boolean = false
    private var outputTensorName: String = ""
    private var normalizedEmbedding: Boolean = false

    suspend fun init(
        modelFilepath: String,
        tokenizerBytes: ByteArray,
        useTokenTypeIds: Boolean,
        outputTensorName: String,
        useFP16: Boolean = false,
        useXNNPack: Boolean = false,
        normalizeEmbeddings: Boolean
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
        ortSession = ortEnvironment.createSession(modelFilepath,options)
        this@SentenceEmbedding.useTokenTypeIds = useTokenTypeIds
        this@SentenceEmbedding.outputTensorName = outputTensorName
        this@SentenceEmbedding.normalizedEmbedding = normalizeEmbeddings
        Log.d(SentenceEmbedding::class.simpleName, "Input Names: " + ortSession.inputNames.toList())
        Log.d(SentenceEmbedding::class.simpleName, "Output Names: " + ortSession.outputNames.toList())
    }

    suspend fun encode(
        sentence: String
    ): FloatArray = withContext(Dispatchers.IO) {
        val result = hfTokenizer.tokenize(sentence)
        val inputTensorMap = mutableMapOf<String,OnnxTensor>()
        val idsTensor =
            OnnxTensor.createTensor(
                ortEnvironment,
                LongBuffer.wrap(result.ids),
                longArrayOf(1, result.ids.size.toLong()),
            )
        inputTensorMap["input_ids"] = idsTensor
        val attentionMaskTensor =
            OnnxTensor.createTensor(
                ortEnvironment,
                LongBuffer.wrap(result.attentionMask),
                longArrayOf(1, result.attentionMask.size.toLong()),
            )
        inputTensorMap["attention_mask"] = attentionMaskTensor
        if (useTokenTypeIds) {
            val tokenTypeIdsTensor =
                OnnxTensor.createTensor(
                    ortEnvironment,
                    LongBuffer.wrap(result.tokenTypeIds),
                    longArrayOf(1, result.tokenTypeIds.size.toLong())
                )
            inputTensorMap["token_type_ids"] = tokenTypeIdsTensor
        }
        val outputs = ortSession.run(inputTensorMap)
        if(normalizedEmbedding) {
            val tokenEmbeddings3D = outputs.get(0).value as Array<Array<FloatArray>> // Change to 3D array

            val tokenEmbeddings = tokenEmbeddings3D[0]

            val pooledEmbeddings = meanPooling(tokenEmbeddings, result.attentionMask)

            val normalizedEmbedding = normalize(pooledEmbeddings)

            return@withContext normalizedEmbedding
        } else {

            val embeddingTensor = outputs.get(outputTensorName).get() as OnnxTensor

            return@withContext embeddingTensor.floatBuffer.array()
        }
    }

    private fun meanPooling(tokenEmbeddings: Array<FloatArray>, attentionMask: LongArray): FloatArray {
        val pooledEmbeddings = FloatArray(tokenEmbeddings[0].size) { 0f }
        var validTokenCount = 0

        for (i in tokenEmbeddings.indices) {
            if (attentionMask[i] == 1L) { // Check if the token is valid
                validTokenCount++
                for (j in tokenEmbeddings[i].indices) {
                    pooledEmbeddings[j] += tokenEmbeddings[i][j]
                }
            }
        }

        // Avoid division by zero
        val divisor = max(validTokenCount, 1)
        for (j in pooledEmbeddings.indices) {
            pooledEmbeddings[j] /= divisor.toFloat()
        }

        return pooledEmbeddings
    }

    // Function to normalize embeddings
    private fun normalize(embeddings: FloatArray): FloatArray {
        // Calculate the L2 norm (Euclidean norm)
        val norm = Math.sqrt(embeddings.sumOf { it * it.toDouble() }).toFloat()

        // Normalize each embedding by dividing by the norm
        return embeddings.map { it / norm }.toFloatArray()
    }

    fun close() {
        ortSession.close()
        ortEnvironment.close()
        hfTokenizer.close()
    }
}
