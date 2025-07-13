package com.example.v1licenta

import CameraFragment
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContentProviderCompat.requireContext
import androidx.core.content.ContextCompat
import com.bumptech.glide.Glide
import com.bumptech.glide.load.engine.DiskCacheStrategy

/**
 * Activitatea principala a aplicatiei care gestioneaza:
 * - Initializarea aplicatiei
 * - Navigarea intre fragmente
 * - Gestionarea permisiunilor
 * - Configurarea initiala a interfetei
 *
 * Functionalitati principale:
 * - Verifica si cere permisiuni necesare
 * - Incarca fragmentul Home la pornire
 * - Gestioneaza schimbarea intre fragmente
 * - Porneste CameraFragment dupa obtinerea permisiunii
 */
class MainActivity : AppCompatActivity() {
    /**
     * Contract pentru cererea de permisiune camera.
     * La primirea raspunsului:
     * - Daca permisiunea este acordata: deschide CameraFragment
     * - Daca permisiunea este refuzata: inchide aplicatia
     */
    private val cameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.container, CameraFragment())
                .commit()
        } else {
            finish()
        }
    }

    /**
     * Metoda apelata la crearea activitatii.
     * Seteaza layout-ul principal si incarca fragmentul Home.
     *
     * @param savedInstanceState Starea salvata (daca exista)
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Start with HomeFragment by default
        if (savedInstanceState == null) {
            supportFragmentManager.beginTransaction()
                .replace(R.id.container, HomeFragment())
                .commit()
        }
    }
}