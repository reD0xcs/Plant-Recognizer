import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import org.tensorflow.lite.support.image.TensorImage
import java.nio.ByteBuffer

/**
 * Clasa responsabila pentru recunoasterea plantelor medicinale folosind TensorFlow Lite.
 *
 * Aceasta clasa ofera functionalitati pentru:
 * - Incarcarea si initializarea modelului de machine learning
 * - Procesarea imaginilor pentru inferenta
 * - Recunoasterea plantelor din imagini
 * - Returnarea rezultatelor cu probabilitati
 *
 * Componente principale:
 * - Interpretor TensorFlow Lite
 * - Procesor de imagini pentru normalizare si redimensionare
 * - Lista de etichete pentru plante
 */
class PlantClassifier(private val context: Context) {
    private var interpreter: Interpreter? = null
    private var labels: List<String> = listOf()

    // Model input/output shapes
    private val inputShape = intArrayOf(1, 224, 224, 3) // Batch, Height, Width, Channels
    private val outputShape = intArrayOf(1, NUM_CLASSES) // Modify NUM_CLASSES to match your model

    // Image processor for basic transformations (additional processing done separately)
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f)) // Normalize to [-1, 1]
        .build()

    // The class labels (your plant labels)
    private val classLabels = arrayOf(
        "Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka", "Ashwagandha", "Avacado", "Bamboo", "Basale", "Betel",
        "Betel_Nut", "Brahmi", "Castor", "Curry_Leaf", "Doddapatre", "Ekka", "Ganike", "Gauva", "Geranium", "Henna",
        "Hibiscus", "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass", "Mango", "Mint", "Nagadali", "Neem", "Nithyapushpa",
        "Nooni", "Pappaya", "Pepper", "Pomegranate", "Raktachandini", "Rose", "Sapota", "Tulasi", "Wood_sorel"
    )

    init {
        initializeInterpreter()
    }

    /**
     * Initializeaza interpretorul TensorFlow Lite si incarca modelul.
     */
    private fun initializeInterpreter() {
        try {
            // Load model
            Log.d("PlantClassifier", "Loading model...")
            val model = FileUtil.loadMappedFile(context, "plant_model.tflite")
            interpreter = Interpreter(model)

            // Load labels (if using a separate file for labels, otherwise you can use the classLabels array directly)
            labels = FileUtil.loadLabels(context, "labels.txt")
            Log.d("PlantClassifier", "Model and labels loaded successfully.")
        } catch (e: Exception) {
            Log.e("PlantClassifier", "Error loading model: ${e.message}")
            e.printStackTrace()
        }
    }

    /**
     * Recunoaste planta dintr-o imagine si returneaza probabilitatile.
     *
     * @param bitmap Imaginea de analizat
     * @return Map<Denumire planta, Probabilitate> cu primele 5 rezultate
     */
    fun recognizePlant(bitmap: Bitmap): Map<String, Float> {
        Log.d("PlantClassifier", "Starting plant recognition.")

        // 1. Apply preprocessing
        val processedMat = ImagePreprocessor.preprocessImage(bitmap)

        // 2. Convert OpenCV Mat to TensorImage
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(processedMat)
        Log.d("PlantClassifier", "TensorImage loaded with shape: ${tensorImage.tensorBuffer.shape.contentToString()}")

        // 3. Apply imageProcessor to resize and normalize
        val processedTensorImage = imageProcessor.process(tensorImage)
        Log.d("PlantClassifier", "Processed TensorImage shape: ${processedTensorImage.tensorBuffer.shape.contentToString()}")

        // 4. Create input buffer
        val inputBuffer = TensorBuffer.createFixedSize(inputShape, DataType.FLOAT32)
        inputBuffer.loadBuffer(processedTensorImage.buffer)

        // 5. Create output buffer
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        // 6. Run inference
        interpreter?.run(inputBuffer.buffer, outputBuffer.buffer.rewind())
        Log.d("PlantClassifier", "Inference completed.")

        // 7. Process results
        val outputs = outputBuffer.floatArray
        val results = mutableMapOf<String, Float>()

        // Map inference results to the corresponding class label
        outputs.forEachIndexed { index, output ->
            results[classLabels[index]] = output
        }

        // Log top results (sorted by confidence)
        Log.d("PlantClassifier", "Top results: ${results.entries.sortedByDescending { it.value }.take(5)}")

        return results.toList()
            .sortedByDescending { (_, value) -> value }
            .take(5) // Get top 5 results
            .toMap()
    }

    /**
     * Elibereaza resursele interpretorului.
     */
    fun close() {
        interpreter?.close()
        Log.d("PlantClassifier", "Interpreter closed.")
    }

    companion object {
        private const val NUM_CLASSES = 40 // The number of classes you have
    }
}
