package com.projects.shubham0204.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding
import com.projects.shubham0204.demo.ui.components.AppProgressDialog
import com.projects.shubham0204.demo.ui.components.hideProgressDialog
import com.projects.shubham0204.demo.ui.components.setProgressDialogText
import com.projects.shubham0204.demo.ui.components.showProgressDialog
import com.projects.shubham0204.demo.ui.theme.SentenceEmbeddingsTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Collections
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    private lateinit var sentenceEmbedding: SentenceEmbedding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        sentenceEmbedding = SentenceEmbedding()

        setContent {
            SentenceEmbeddingsTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->

                    var isModelLoaded by remember{ mutableStateOf(false) }

                    LaunchedEffect(0) {
                        val modelBytes = assets.open("all-MiniLM-L6-V2.onnx").readBytes()
                        val tokenizerBytes = copyToLocalStorage()
                        sentenceEmbedding.init(
                            modelBytes,
                            tokenizerBytes,
                            useXNNPack = true
                        )
                        isModelLoaded = true
                    }

                    if (!isModelLoaded) {
                        showProgressDialog()
                        setProgressDialogText("Loading model...")
                    }
                    else {
                        hideProgressDialog()
                    }

                    Column(modifier= Modifier
                        .padding(innerPadding)
                        .padding(horizontal = 16.dp)
                        .padding(top = 16.dp)) {

                        var sentence1 by remember{ mutableStateOf("What is the population of London?") }
                        var sentence2 by remember{ mutableStateOf("Delhi has a population of 32 million") }
                        var cosineSimilarity by remember{ mutableStateOf<Float?>(null) }
                        var inferenceTime by remember{ mutableStateOf<Long?>(null) }

                        Text(
                            text = "Using all-MiniLM-L6-V2 from 🤗 sentence-transformers",
                            style = MaterialTheme.typography.headlineSmall
                        )
                        Spacer(modifier = Modifier.height(24.dp))

                        TextField(
                            modifier = Modifier.fillMaxWidth() ,
                            value = sentence1,
                            onValueChange = { sentence1 = it },
                            placeholder = { Text(text = "Enter first sentence...")}
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        TextField(
                            modifier = Modifier.fillMaxWidth() ,
                            value = sentence2,
                            onValueChange = { sentence2 = it },
                            placeholder = { Text(text = "Enter second sentence...")}
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(onClick = {
                            CoroutineScope(Dispatchers.Default).launch {
                                val results = predict(sentence1, sentence2)
                                withContext(Dispatchers.Main) {
                                    cosineSimilarity = results.first
                                    inferenceTime = results.second
                                }
                            }

                        }) {
                            Text(text = "Calculate Similarity")
                        }
                        Spacer(modifier = Modifier.height(24.dp))
                        if (cosineSimilarity != null) {
                            Text(text = "Inference time (millis): $inferenceTime ms" )
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        if (cosineSimilarity != null) {
                            LinearProgressIndicator(
                                modifier = Modifier.fillMaxWidth(),
                                progress = ((cosineSimilarity?:0.0f) + 1.0f) / 2f
                            )
                            Text(
                                text = cosineSimilarity.toString(),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }
                        AppProgressDialog()
                    }

                }
            }
        }
    }

    private suspend fun predict(
        sentence1: String,
        sentence2: String
    ): Pair<Float,Long> = withContext(Dispatchers.IO) {
        showProgressDialog()
        setProgressDialogText("⚡ Encoding sentence 1...")
        val t1 = System.currentTimeMillis()
        val sentenceEmbeddings = Collections.synchronizedList(mutableListOf<FloatArray>())
        listOf(
            launch { sentenceEmbeddings.add(sentenceEmbedding.encode(sentence1)) } ,
            launch { sentenceEmbeddings.add(sentenceEmbedding.encode(sentence2)) }
        ).joinAll()
        val cosineSimilarity = cosineDistance(sentenceEmbeddings[0],sentenceEmbeddings[1])
        val inferenceTime = System.currentTimeMillis() - t1
        hideProgressDialog()
        return@withContext Pair(cosineSimilarity,inferenceTime)
    }

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

    private fun copyToLocalStorage(): ByteArray {
        val tokenizerBytes = assets.open("tokenizer.json").readBytes()
        val storageFile = File(filesDir, "tokenizer.json")
        if (!storageFile.exists()) {
            storageFile.writeBytes(tokenizerBytes)
        }
        return storageFile.readBytes()
    }

}
