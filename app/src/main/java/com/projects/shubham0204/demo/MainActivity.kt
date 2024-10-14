package com.projects.shubham0204.demo

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
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
import java.io.FileOutputStream
import java.util.Collections
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    private lateinit var sentenceEmbedding: SentenceEmbedding
    private val modelConfigState = mutableStateOf<ModelConfig?>(null)
    private val showChooseModelDialogState = mutableStateOf(true)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        sentenceEmbedding = SentenceEmbedding()

        setContent {
            SentenceEmbeddingsTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->





                    Column(modifier= Modifier
                        .padding(innerPadding)
                        .padding(horizontal = 16.dp)
                        .padding(top = 16.dp)) {

                        var isModelLoaded by remember{ mutableStateOf(false) }
                        val modelConfig by remember{ modelConfigState }

                        var sentence1 by remember{ mutableStateOf("What is the population of London?") }
                        var sentence2 by remember{ mutableStateOf("Delhi has a population of 32 million") }
                        var cosineSimilarity by remember{ mutableStateOf<Float?>(null) }
                        var inferenceTime by remember{ mutableStateOf<Long?>(null) }

                        modelConfig?.let { config ->
                            CoroutineScope(Dispatchers.IO).launch {
                                if (!isModelLoaded) {
                                    sentenceEmbedding.init(
                                        copyAndReturnPath(config.modelAssetsFilepath),
                                        copyAndReturnBytes(config.tokenizerAssetsFilepath),
                                        useTokenTypeIds = config.useTokenTypeIds,
                                        outputTensorName = config.outputTensorName,
                                        normalizeEmbeddings = config.normalizeEmbeddings
                                    )
                                    isModelLoaded = true
                                }
                            }
                        }

                        if (isModelLoaded) {
                            Text(
                                text = "Using ${modelConfig?.modelName} from 🤗 sentence-transformers",
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
                            Row {
                                Button(
                                    modifier = Modifier
                                        .padding(4.dp)
                                        .fillMaxWidth()
                                        .weight(1f),
                                    onClick = {
                                    CoroutineScope(Dispatchers.Default).launch {
                                        val results = predict(sentence1, sentence2)
                                        withContext(Dispatchers.Main) {
                                            cosineSimilarity = results.first
                                            inferenceTime = results.second
                                        }
                                    }
                                }) {
                                    Text(text = "Similarity Score")
                                }
                                Button(
                                    modifier = Modifier
                                        .padding(4.dp)
                                        .fillMaxWidth()
                                        .weight(1f),
                                    onClick = {
                                        modelConfigState.value = null
                                        cosineSimilarity = null
                                        isModelLoaded = false
                                        showChooseModelDialogState.value = true
                                }) {
                                    Text(text = "Choose model")
                                }
                            }

                        }

                        cosineSimilarity?.let {
                            Spacer(modifier = Modifier.height(24.dp))
                            Text(text = "Inference time (millis): $inferenceTime ms" )
                            Spacer(modifier = Modifier.height(8.dp))
                            LinearProgressIndicator(
                                modifier = Modifier.fillMaxWidth(),
                                progress = (it + 1.0f) / 2f
                            )
                            Text(
                                text = it.toString(),
                                style = MaterialTheme.typography.labelSmall
                            )
                        }

                        AppProgressDialog()
                        ChooseModelDialog()
                    }

                }
            }
        }
    }

    @Composable
    fun ChooseModelDialog() {
        var showDialog by remember { showChooseModelDialogState }
        val modelConfigs = Model.entries.map{ getModelConfig(it) }
        if (showDialog) {
            Dialog(onDismissRequest = { /** Non cancellable dialog **/ }) {
                Column(modifier = Modifier
                    .padding(16.dp)
                    .background(Color.White, RoundedCornerShape(16.dp))
                    .padding(24.dp)
                    .fillMaxWidth()
                ) {
                    Text(text = "Choose model", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                    Spacer(modifier = Modifier.height(8.dp))
                    LazyColumn {
                        items(modelConfigs) {
                            Text(
                                modifier = Modifier
                                    .clickable {
                                        showDialog = false
                                        modelConfigState.value = it
                                    }
                                    .padding(16.dp)
                                    .fillMaxWidth(),
                                text = it.modelName,
                            )
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        sentenceEmbedding.close()
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
            //launch { sentenceEmbeddings.add(sentenceEmbedding.encode(sentence2)) }
        ).joinAll()

        Log.d("Size", sentenceEmbeddings[0].size.toString())
        Log.d("Size", sentenceEmbeddings[0].contentToString())
        //val cosineSimilarity = cosineDistance(sentenceEmbeddings[0],sentenceEmbeddings[1])
        val inferenceTime = System.currentTimeMillis() - t1
        hideProgressDialog()
        return@withContext Pair(1.0f,inferenceTime)
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

    // Copy the file from the assets to the app's internal/private storage
    // and return its absolute path
    private fun copyAndReturnPath(assetsFilepath: String): String {
        val storageFile = File(filesDir, assetsFilepath)
        if (!storageFile.exists()) {
            storageFile.parentFile?.mkdir()
            FileOutputStream(storageFile).use { outputStream ->
                assets.open(assetsFilepath).use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return storageFile.absolutePath
    }

    // Copy the file from the assets to the app's internal/private storage
    // and return its data as a ByteArray
    private fun copyAndReturnBytes(assetsFilepath: String): ByteArray {
        val storageFile = File(filesDir, assetsFilepath)
        if (!storageFile.exists()) {
            storageFile.parentFile?.mkdir()
            FileOutputStream(storageFile).use { outputStream ->
                assets.open(assetsFilepath).use { inputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        return storageFile.readBytes()
    }

}
