package com.example.v1licenta

import android.app.Application
import org.opencv.android.OpenCVLoader

class PlantRecognizerApp : Application() {
    override fun onCreate() {
        super.onCreate()
        initOpenCV()
    }

    private fun initOpenCV() {
        if (!OpenCVLoader.initDebug()) {
            println("OpenCV failed to load")
        } else {
            println("OpenCV loaded successfully")
        }
    }
}