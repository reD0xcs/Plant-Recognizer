package com.example.v1licenta

import CameraFragment
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.core.app.ActivityCompat

/**
 * Fragment principal care serveste ca pagina de start a aplicatiei.
 *
 * Acest fragment ofera functionalitatile:
 * - Navigare catre CameraFragment
 * - Navigare catre HistoryFragment
 * - Gestionare permisiuni camera
 *
 * Contine butoane pentru:
 * - Deschiderea camerei (cu verificare permisiune)
 * - Accesarea istoricului recunoasterilor
 */
class HomeFragment : Fragment() {
    /**
     * Creeaza view-ul fragmentului folosind layout-ul definit in R.layout.fragment_home
     */
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_home, container, false)
    }

    /**
     * Configurarea view-ului dupa creare
     * Seteaza actiunile pentru butoanele principale:
     * - Buton camera (cu verificare permisiuni)
     * - Buton istoric (navigare simpla)
     */
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Camera button click
        view.findViewById<Button>(R.id.btnCamera).setOnClickListener {
            navigateToCamera()
        }

        // History button click (placeholder)
        view.findViewById<Button>(R.id.btnHistory).setOnClickListener {
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, HistoryFragment())
                .addToBackStack("home")
                .commit()
        }
    }

    /**
     * Navigheaza catre CameraFragment daca permisiunea este acordata.
     * Daca nu, cere permisiunea pentru camera.
     */
    private fun navigateToCamera() {
        // Check camera permission first
        if (ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            // Permission granted, open camera
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, CameraFragment())
                .addToBackStack("home")
                .commit()
        } else {
            // Request permission
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    /**
     * Proceseaza rezultatul cererii de permisiuni
     * Daca permisiunea este acordata, navigheaza catre camera
     * Daca este refuzata, afiseaza un mesaj informativ
     */
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            navigateToCamera()
        } else {
            Toast.makeText(
                requireContext(),
                "Camera permission required to use this feature",
                Toast.LENGTH_SHORT
            ).show()
        }
    }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 100
    }
}