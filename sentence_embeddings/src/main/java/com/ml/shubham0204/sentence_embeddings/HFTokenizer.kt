package com.ml.shubham0204.sentence_embeddings

import org.json.JSONObject

class HFTokenizer(private val tokenizerFilepath: String) {

    data class Result(
        val ids: LongArray = longArrayOf(),
        val attentionMask: LongArray = longArrayOf()
    )

    private val tokenizerPtr: Long = createTokenizer(tokenizerFilepath)

    fun tokenize(text: String): Result {
        val output = tokenize(tokenizerPtr, text)
        val jsonObject = JSONObject(output)
        val idsArray = jsonObject.getJSONArray("ids")
        val ids = LongArray(idsArray.length())
        for (i in 0 until idsArray.length()) {
            ids[i] = (idsArray.get(i) as Int).toLong()
        }
        val attentionMaskArray = jsonObject.getJSONArray("attention_mask")
        val attentionMask = LongArray(attentionMaskArray.length())
        for (i in 0 until attentionMaskArray.length()) {
            attentionMask[i] = (attentionMaskArray.get(i) as Int).toLong()
        }
        return Result(ids, attentionMask)
    }

    fun close() {
        deleteTokenizer(tokenizerPtr)
    }

    private external fun createTokenizer(tokenizerFilepath: String): Long

    private external fun tokenize(tokenizerPtr: Long, text: String): String

    private external fun deleteTokenizer(tokenizerPtr: Long)

    companion object {
        init {
            System.loadLibrary("hftokenizer")
        }
    }
}
