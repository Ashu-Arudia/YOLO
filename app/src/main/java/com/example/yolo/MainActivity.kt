package com.example.yolo

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : AppCompatActivity() {

    private lateinit var tvStatus: TextView
    private lateinit var tvBackendInfo: TextView
    private lateinit var btnRefresh: Button

    // Default backend URL when running on Android emulator (10.0.2.2 maps to host machine localhost)
    private val backendBaseUrl = "http://10.0.2.2:8000"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus = findViewById(R.id.tvStatus)
        tvBackendInfo = findViewById(R.id.tvBackendInfo)
        btnRefresh = findViewById(R.id.btnRefresh)

        btnRefresh.setOnClickListener {
            checkBackendHealth()
        }

        checkBackendHealth()
    }

    private fun checkBackendHealth() {
        tvStatus.text = "Checking SportsVision AI Backend..."
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val url = URL("$backendBaseUrl/api/v1/health")
                val connection = (url.openConnection() as HttpURLConnection).apply {
                    requestMethod = "GET"
                    connectTimeout = 5000
                    readTimeout = 5000
                }

                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    val json = JSONObject(response)
                    val service = json.optString("service", "SportsVision AI")
                    val status = json.optString("status", "unknown")
                    val device = json.optString("device", "cpu")
                    val torchVer = json.optString("torch_version", "unknown")

                    withContext(Dispatchers.Main) {
                        tvStatus.text = "Connected ($status)"
                        tvBackendInfo.text = "Service: $service\nDevice: $device\nPyTorch: $torchVer"
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        tvStatus.text = "Server responded with HTTP $responseCode"
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Connection failed: ${e.localizedMessage ?: "Unknown error"}"
                    tvBackendInfo.text = "Note: Ensure the backend is running at http://localhost:8000."
                }
            }
        }
    }
}
