package com.projects.shubham0204.demo_model2vec

import android.content.ContentResolver
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.ml.shubham0204.model2vec.Model2Vec
import com.projects.shubham0204.demo_model2vec.ui.components.AppProgressDialog
import com.projects.shubham0204.demo_model2vec.ui.theme.SentenceEmbeddingsTheme
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import kotlin.math.pow
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {
    private lateinit var model2vec: Model2Vec

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        model2vec =
            Model2Vec(
                embeddingsPath = copyAndReturnPath("embeddings.safetensors"),
                tokenizerPath = copyAndReturnPath("tokenizer.json"),
                numThreads = 2,
            )
        setContent {
            SentenceEmbeddingsTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier =
                            Modifier
                                .padding(innerPadding)
                                .padding(horizontal = 16.dp)
                                .padding(top = 16.dp),
                    ) {
                        var fileName1 by remember { mutableStateOf("No file selected") }
                        var wordCount1 by remember { mutableStateOf(0) }
                        var fileName2 by remember { mutableStateOf("No file selected") }
                        var wordCount2 by remember { mutableStateOf(0) }
                        var cosineSimilarity by remember { mutableStateOf<Float?>(null) }
                        var inferenceTime by remember { mutableStateOf<Long?>(null) }
                        var text1 by remember { mutableStateOf("") }
                        var text2 by remember { mutableStateOf("") }

                        val filePickerLauncher1 =
                            rememberLauncherForActivityResult(
                                contract = ActivityResultContracts.OpenDocument(),
                                onResult = { uri: Uri? ->
                                    uri?.let {
                                        val (arr, count) = readFileContent(uri)
                                        fileName1 = arr[0]
                                        text1 = arr[1]
                                        wordCount1 = count
                                    }
                                },
                            )

                        val filePickerLauncher2 =
                            rememberLauncherForActivityResult(
                                contract = ActivityResultContracts.OpenDocument(),
                                onResult = { uri: Uri? ->
                                    uri?.let {
                                        val (arr, count) = readFileContent(uri)
                                        fileName2 = arr[0]
                                        text2 = arr[1]
                                        wordCount2 = count
                                    }
                                },
                            )

                        Text(
                            text = "Using model2vec from 🤗 sentence-transformers",
                            style = MaterialTheme.typography.headlineSmall,
                        )
                        Spacer(modifier = Modifier.height(24.dp))

                        FileCard(
                            onFileSelected = { filePickerLauncher1.launch(arrayOf("text/plain")) },
                            fileName = fileName1,
                            wordCount = wordCount1,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        FileCard(
                            onFileSelected = { filePickerLauncher2.launch(arrayOf("text/plain")) },
                            fileName = fileName2,
                            wordCount = wordCount2,
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row {
                            Button(
                                modifier =
                                    Modifier
                                        .padding(4.dp)
                                        .fillMaxWidth()
                                        .weight(1f),
                                onClick = {
                                    val results = predict(text1, text2)
                                    cosineSimilarity = results.first
                                    inferenceTime = results.second
                                },
                            ) {
                                Text(text = "Similarity Score")
                            }
                        }

                        cosineSimilarity?.let {
                            Spacer(modifier = Modifier.height(24.dp))
                            Text(text = "Inference time (millis): $inferenceTime ms")
                            Spacer(modifier = Modifier.height(8.dp))
                            LinearProgressIndicator(
                                progress = {
                                    (it + 1.0f) / 2f
                                },
                                modifier = Modifier.fillMaxWidth(),
                            )
                            Text(
                                text = it.toString(),
                                style = MaterialTheme.typography.labelSmall,
                            )
                        }

                        AppProgressDialog()
                    }
                }
            }
        }
    }

    @Composable
    fun FileCard(
        onFileSelected: () -> Unit,
        fileName: String,
        wordCount: Int,
    ) {
        Card(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Button(onClick = onFileSelected) {
                    Text(text = "Select File")
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(text = "File: $fileName")
                Spacer(modifier = Modifier.height(8.dp))
                Text(text = "Word Count: $wordCount")
            }
        }
    }

    private fun predict(
        sentence1: String,
        sentence2: String,
    ): Pair<Float, Long> {
        val t1 = System.currentTimeMillis()
        val sentenceEmbeddings = model2vec.encode(listOf(sentence1, sentence2))
        val cosineSimilarity = cosineDistance(sentenceEmbeddings[0], sentenceEmbeddings[1])
        val inferenceTime = System.currentTimeMillis() - t1
        return Pair(cosineSimilarity, inferenceTime)
    }

    private fun cosineDistance(
        x1: FloatArray,
        x2: FloatArray,
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

    private fun readFileContent(uri: Uri): Pair<Array<String>, Int> {
        val inputStream = contentResolver.openInputStream(uri)
        val reader = BufferedReader(InputStreamReader(inputStream))
        val content = reader.readText()
        val wordCount = content.split("\\s+".toRegex()).size
        val fileName = queryName(contentResolver, uri)
        return Pair(arrayOf(fileName, content), wordCount)
    }

    private fun queryName(
        resolver: ContentResolver,
        uri: Uri,
    ): String {
        val projection = arrayOf(OpenableColumns.DISPLAY_NAME)
        val returnCursor = resolver.query(uri, projection, null, null, null)
        assert(returnCursor != null)
        returnCursor?.moveToFirst()
        val name = returnCursor?.getString(0) ?: "Unknown"
        returnCursor?.close()
        return name
    }

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
}
