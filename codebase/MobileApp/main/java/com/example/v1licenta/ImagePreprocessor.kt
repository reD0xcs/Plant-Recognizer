import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Obiect pentru preprocesarea imaginilor inainte de recunoasterea plantelor.
 *
 * Acest obiect contine metode pentru:
 * - Conversia si pregatirea imaginilor pentru modelul de machine learning
 * - Aplicarea unor operatii de imbunatatire a calitatii imaginii
 * - Normalizarea dimensiunilor imaginilor
 *
 * Functionalitati principale:
 * - Conversie Bitmap -> Mat (OpenCV)
 * - Padding pentru imagini nesquare
 * - Redimensionare la 224x224 px
 * - Eliminare umbre (CLAHE pe canalul S)
 * - Reducere zgomot (filtru bilateral)
 */
object ImagePreprocessor {
    /**
     * Preproceseaza o imagine pentru recunoasterea plantelor.
     *
     * @param bitmap Imaginea sursa ca Bitmap
     * @return Bitmap preprocesat, pregatit pentru modelul ML
     */
    fun preprocessImage(bitmap: Bitmap): Bitmap {
        Log.d("ImagePreprocessor", "Starting image preprocessing.")

        // Convert Bitmap to Mat
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        Log.d("ImagePreprocessor", "Bitmap converted to Mat: ${mat.rows()}x${mat.cols()}")

        // Convert grayscale to BGR if needed
        if (mat.channels() == 1) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2BGR)
            Log.d("ImagePreprocessor", "Converted grayscale to BGR.")
        } else if (mat.channels() == 4) {
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGRA2BGR)
            Log.d("ImagePreprocessor", "Converted BGRA to BGR.")
        }

        // Pad to square if needed
        val height = mat.rows()
        val width = mat.cols()
        if (height != width) {
            val size = maxOf(height, width)
            val padTop = (size - height) / 2
            val padBottom = size - height - padTop
            val padLeft = (size - width) / 2
            val padRight = size - width - padLeft
            Core.copyMakeBorder(
                mat, mat, padTop, padBottom, padLeft, padRight,
                Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0)
            )
            Log.d("ImagePreprocessor", "Image padded to square: ${mat.rows()}x${mat.cols()}")
        }

        // Resize to 224x224
        Imgproc.resize(mat, mat, Size(224.0, 224.0))
        Log.d("ImagePreprocessor", "Image resized to 224x224.")

        // Shadow removal (CLAHE on HSV)
        val hsv = Mat()
        Imgproc.cvtColor(mat, hsv, Imgproc.COLOR_BGR2HSV)
        val hsvChannels = mutableListOf<Mat>()
        Core.split(hsv, hsvChannels)

        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(hsvChannels[1], hsvChannels[1]) // S channel
        Log.d("ImagePreprocessor", "CLAHE applied on S channel.")

        Core.merge(hsvChannels, hsv)
        Imgproc.cvtColor(hsv, mat, Imgproc.COLOR_HSV2BGR)

        // Denoise with bilateral filter
        val filtered = Mat()
        Imgproc.bilateralFilter(mat, filtered, 9, 75.0, 75.0)
        Log.d("ImagePreprocessor", "Denoised image with bilateral filter.")

        // Convert back to Bitmap (normalization is handled by TensorImage)
        val resultBitmap = Bitmap.createBitmap(filtered.cols(), filtered.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(filtered, resultBitmap)

        Log.d("ImagePreprocessor", "Image preprocessing completed.")
        return resultBitmap
    }
}
