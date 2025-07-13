import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.GestureDetector
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.bumptech.glide.Glide
import com.bumptech.glide.load.engine.DiskCacheStrategy
import com.bumptech.glide.request.RequestOptions
import com.example.v1licenta.HistoryFragment
import com.example.v1licenta.MedicinalPlant
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream
import java.io.File
import com.example.v1licenta.R
import java.io.FileOutputStream

/**
 * Fragment pentru gestionarea functionalitatilor de camera si recunoasterea plantelor.
 *
 * Acest fragment ofera urmatoarele functionalitati:
 * - Capturarea de fotografii folosind camera dispozitivului
 * - Selectarea de imagini din galerie
 * - Recunoasterea plantelor medicinale din imagini
 * - Afisarea detaliilor despre plante
 * - Salvarea rezultatelor in istoric
 *
 * Foloseste:
 * - CameraX API pentru operatiuni cu camera
 * - TensorFlow Lite pentru modelul ML de recunoastere a plantelor
 * - Detectare de gesturi pentru navigare
 * - Bottom sheets pentru afisarea detaliilor plantelor
 */
class CameraFragment : Fragment() {
    private lateinit var plantClassifier: PlantClassifier
    private lateinit var imageCapture: ImageCapture
    private val CAMERA_PERMISSION_REQUEST_CODE = 1001
    private lateinit var plantsList: List<MedicinalPlant>
    private lateinit var gestureDetector: GestureDetector

    private val openGalleryLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            uri?.let {
                val bitmap = MediaStore.Images.Media.getBitmap(requireContext().contentResolver, uri)
                val preprocessedBitmap = ImagePreprocessor.preprocessImage(bitmap)
                classifyImage(preprocessedBitmap)
            }
        }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_camera, container, false)
    }

    private fun openHistoryFragment() {
        parentFragmentManager.beginTransaction()
            .setCustomAnimations(
                R.anim.slide_in_left,  // enter
                R.anim.slide_out_right, // exit
                R.anim.slide_in_right,  // popEnter (when coming back)
                R.anim.slide_out_left   // popExit (when going back)
            )
            .replace(R.id.container, HistoryFragment())
            .addToBackStack(null)
            .setReorderingAllowed(true) // important for animations
            .commit()
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        plantClassifier = PlantClassifier(requireContext())
        loadPlantsData()

        gestureDetector = GestureDetector(requireContext(), object : GestureDetector.SimpleOnGestureListener() {
            override fun onFling(
                e1: MotionEvent?,
                e2: MotionEvent,
                velocityX: Float,
                velocityY: Float
            ): Boolean {
                if (e1 == null || e2 == null) return false

                val diffX = e2.x - e1.x
                val diffY = e2.y - e1.y
                if (Math.abs(diffX) > Math.abs(diffY)) {
                    if (Math.abs(diffX) > 100 && Math.abs(velocityX) > 100) {
                        if (diffX > 0) {
                            openHistoryFragment()
                            return true
                        }
                    }
                }
                return false
            }
        })

        //Set touch listener AFTER gestureDetector is initialized
        view.setOnTouchListener { _, event ->
            gestureDetector.onTouchEvent(event)
            true
        }

        view.findViewById<ImageButton>(R.id.backButton)?.setOnClickListener {
            parentFragmentManager.popBackStack()
        }

        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST_CODE)
        } else {
            startCamera()
        }

        view.findViewById<View>(R.id.captureButton).setOnClickListener {
            takePicture()
        }

        view.findViewById<View>(R.id.galleryButton).setOnClickListener {
            openGallery()
        }
    }

    private fun loadPlantsData() {
        try {
            val inputStream: InputStream = requireContext().assets.open("medicinal_plants.json")
            val size = inputStream.available()
            val buffer = ByteArray(size)
            inputStream.read(buffer)
            inputStream.close()
            val json = String(buffer, Charsets.UTF_8)

            val gson = Gson()
            val listType = object : TypeToken<List<MedicinalPlant>>() {}.type
            plantsList = gson.fromJson(json, listType)
        } catch (e: Exception) {
            Log.e("CameraFragment", "Error loading plant data: ${e.message}")
            plantsList = emptyList()
        }
    }

    private fun startCamera() {
        val previewView = view?.findViewById<PreviewView>(R.id.previewView) ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder().build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("CameraFragment", "Error binding camera: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun takePicture() {
        val photoFile = File(requireContext().externalMediaDirs.first(), "photo.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(requireContext()),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    val preprocessedBitmap = ImagePreprocessor.preprocessImage(bitmap)
                    classifyImage(preprocessedBitmap)
                }

                override fun onError(exception: ImageCaptureException) {
                    view?.findViewById<TextView>(R.id.resultTextView)?.text =
                        "Error capturing image: ${exception.message}"
                }
            }
        )
    }

    private fun openGallery() {
        openGalleryLauncher.launch("image/*")
    }

    private fun classifyImage(bitmap: Bitmap) {
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                plantClassifier.recognizePlant(bitmap)
            }
            updateResults(result)
        }
    }

    private fun updateResults(results: Map<String, Float>) {
        activity?.runOnUiThread {
            val topResult = results.entries.firstOrNull()
            val resultText = topResult?.let {
                "${it.key} (${String.format("%.1f", it.value * 100)}%)"
            } ?: "No results"

            view?.findViewById<TextView>(R.id.resultTextView)?.text = resultText

            topResult?.let { result ->
                val plant = plantsList.firstOrNull { it.name.equals(result.key, ignoreCase = true) }
                if (plant != null) {
                    val photoFile = File(requireContext().externalMediaDirs.first(), "photo.jpg")
                    val bitmap = BitmapFactory.decodeFile(photoFile.absolutePath)
                    showConfirmationDialog(bitmap, plant)
                }
            }

        }
    }

    private fun showPlantDetails(plantName: String) {
        val plant = plantsList.firstOrNull { it.name.equals(plantName, ignoreCase = true) } ?: return

        val bottomSheetView = LayoutInflater.from(requireContext()).inflate(
            R.layout.plant_details_bottom_sheet,
            null,
            false
        )

        val bottomSheet = BottomSheetDialog(requireContext())
        bottomSheet.setContentView(bottomSheetView)

        // Set plant data
        bottomSheetView.findViewById<TextView>(R.id.plantName).text = plant.name
        bottomSheetView.findViewById<TextView>(R.id.scientificName).text = plant.scientific_name
        bottomSheetView.findViewById<TextView>(R.id.plantUses).text = plant.uses.joinToString(", ")
        bottomSheetView.findViewById<TextView>(R.id.plantPartsUsed).text = plant.parts_used.joinToString(", ")
        bottomSheetView.findViewById<TextView>(R.id.plantPreparation).text = plant.preparation.joinToString(", ")
        bottomSheetView.findViewById<TextView>(R.id.plantBenefits).text = plant.benefits.joinToString(", ")

        // Load image
        try {
            when {
                plant.photo_url.isEmpty() -> {
                    loadPlaceholder(bottomSheetView.findViewById(R.id.plantImage))
                }
                plant.photo_url.startsWith("http") -> {
                    Glide.with(this)
                        .load(plant.photo_url)
                        .apply(createGlideOptions())
                        .into(bottomSheetView.findViewById(R.id.plantImage))
                }
                else -> {
                    // For local assets - NEW IMPLEMENTATION
                    val inputStream = requireContext().assets.open(plant.photo_url)
                    val bytes = inputStream.readBytes()
                    Glide.with(this)
                        .load(bytes)
                        .apply(createGlideOptions())
                        .into(bottomSheetView.findViewById(R.id.plantImage))
                }
            }
        } catch (e: Exception) {
            Log.e("PlantDetails", "Error loading image: ${e.message}")
            loadPlaceholder(bottomSheetView.findViewById(R.id.plantImage))
        }

        bottomSheet.show()
    }

    // Helper function for Glide options
    private fun createGlideOptions(): RequestOptions {
        return RequestOptions()
            .placeholder(R.drawable.plant_placeholder)
            .error(R.drawable.plant_placeholder)
            .diskCacheStrategy(DiskCacheStrategy.ALL)
    }

    // Helper function for placeholder
    private fun loadPlaceholder(imageView: ImageView) {
        Glide.with(this)
            .load(R.drawable.plant_placeholder)
            .into(imageView)
    }

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
            startCamera()
        } else {
            view?.findViewById<TextView>(R.id.resultTextView)?.text = "Camera permission denied"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        plantClassifier.close()
    }

    private fun showConfirmationDialog(bitmap: Bitmap, plant: MedicinalPlant) {
        val dialogView = LayoutInflater.from(requireContext()).inflate(R.layout.dialog_confirm_photo, null)
        val dialog = AlertDialog.Builder(requireContext())
            .setView(dialogView)
            .setCancelable(false)
            .create()

        dialogView.findViewById<ImageView>(R.id.capturedImage).setImageBitmap(bitmap)
        dialogView.findViewById<TextView>(R.id.plantName).text = plant.name

        dialogView.findViewById<Button>(R.id.btnConfirm).setOnClickListener {
            saveToHistory(bitmap, plant)
            dialog.dismiss()
            showPlantDetails(plant.name)
        }

        dialogView.findViewById<Button>(R.id.btnRetake).setOnClickListener {
            dialog.dismiss()
        }

        dialog.show()
    }

    private fun saveToHistory(bitmap: Bitmap, plant: MedicinalPlant) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Save image to internal storage
                val imagesDir = File(requireContext().filesDir, "plant_images")
                if (!imagesDir.exists()) imagesDir.mkdirs()

                val imageFile = File(imagesDir, "${System.currentTimeMillis()}.jpg")
                FileOutputStream(imageFile).use { out ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
                }

                // Add to history
                HistoryFragment.addToHistory(
                    requireContext(),
                    plant,
                    imageFile.absolutePath
                )

                withContext(Dispatchers.Main) {
                    Toast.makeText(requireContext(), "Saved to history", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                Log.e("CameraFragment", "Error saving to history", e)
            }
        }
    }
}