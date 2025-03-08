package com.ml.shubham0204.model2vec

class Model2Vec(
    embeddingsPath: String,
    tokenizerPath: String,
    numThreads: Int,
) {
    private var nativeHandle: Long = 0L

    companion object {
        init {
            System.loadLibrary("model2vec")
        }
    }

    init {
        nativeHandle = create(embeddingsPath, tokenizerPath, numThreads)
    }

    fun encode(sequences: List<String>): List<FloatArray> {
        sequences.forEach { addSeqBuffer(nativeHandle, it) }
        val embeddings = encode(nativeHandle, Runtime.getRuntime().availableProcessors())
        clearSeqBuffer(nativeHandle)
        return embeddings.toList()
    }

    protected fun finalize() {
        release(nativeHandle)
    }

    private external fun create(
        embeddingsPath: String,
        tokenizerPath: String,
        numThreads: Int,
    ): Long

    private external fun addSeqBuffer(
        handle: Long,
        sequence: String,
    )

    private external fun clearSeqBuffer(handle: Long)

    private external fun encode(
        handle: Long,
        numThreads: Int,
    ): Array<FloatArray>

    private external fun release(handle: Long)
}
