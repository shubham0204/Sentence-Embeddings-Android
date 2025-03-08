package com.ml.shubham0204.model2vec

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class Model2VecTest {
    /**
     * Paths to the embeddings and tokenizer files
     * Download model.safetensors and tokenizer.json from this HF repository,
     * https://huggingface.co/minishlab/potion-base-8M/tree/main
     *
     * Upload `embeddings.safetensors` and `tokenizer.json` to the test-device using,
     *
     * adb push model.safetensors /data/local/tmp/embeddings.safetensors
     * adb push tokenizer.json /data/local/tmp/tokenizer.json
     */
    private val embeddingsPath = "/data/local/tmp/embeddings.safetensors"
    private val tokenizerPath = "/data/local/tmp/tokenizer.json"
    private val embeddingDims = 256

    @Test
    fun encode_works() {
        val model2Vec = Model2Vec(embeddingsPath, tokenizerPath, 4)
        val embeddings = model2Vec.encode(listOf("Hello World"))
        val embeddings2 = model2Vec.encode(listOf("Hello World", "Hello"))
        assert(embeddings.size == 1)
        assert(embeddings[0].size == embeddingDims)
    }
}
