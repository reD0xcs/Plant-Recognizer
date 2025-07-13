package com.example.v1licenta

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import android.net.Uri
import android.util.Log
import android.view.GestureDetector
import android.view.MotionEvent
import java.io.File
import java.io.InputStream

/**
 * Fragment pentru afisarea istoricului de recunoastere a plantelor.
 *
 * Acest fragment ofera urmatoarele functionalitati:
 * - Afisarea cronologica a recunoasterilor anterioare
 * - Incarcarea si stocarea istoricului in SharedPreferences
 * - Gestioneaza stergerea elementelor din istoric
 * - Afiseaza detalii despre plante prin bottom sheet
 * - Navigare prin gesturi (swipe)
 *
 * Componente principale:
 * - RecyclerView pentru lista de elemente
 * - Adaptor personalizat pentru afisare
 * - BottomSheetDialog pentru detalii
 * - GestureDetector pentru navigare
 */
class HistoryFragment : Fragment() {
    private val historyItems = mutableListOf<HistoryItem>()
    private lateinit var adapter: HistoryAdapter
    private lateinit var plantsList: List<MedicinalPlant>
    private lateinit var gestureDetector: GestureDetector

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_history, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        loadPlantsData()
        loadHistory()

        val recyclerView = view.findViewById<RecyclerView>(R.id.historyRecyclerView)
        adapter = HistoryAdapter(
            historyItems,
            onClick = { item -> showPlantDetails(item) },
            onDeleteClick = { item, position -> deleteHistoryItem(position) }
        )

        recyclerView.layoutManager = LinearLayoutManager(requireContext())
        recyclerView.adapter = adapter

        view.findViewById<ImageButton>(R.id.backButton).setOnClickListener {
            parentFragmentManager.popBackStack()
        }
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
                        if (diffX < 0) {  // swipe left
                            parentFragmentManager.popBackStack()
                            return true
                        }
                    }
                }
                return false
            }
        })

        recyclerView.addOnItemTouchListener(object : RecyclerView.SimpleOnItemTouchListener() {
            override fun onInterceptTouchEvent(rv: RecyclerView, e: MotionEvent): Boolean {
                return gestureDetector.onTouchEvent(e)
            }
        })

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
            Log.e("HistoryFragment", "Error loading plant data: ${e.message}")
            plantsList = emptyList()
        }
    }

    private fun loadHistory() {
        val sharedPref = requireContext().getSharedPreferences("PlantHistory", Context.MODE_PRIVATE)
        val gson = Gson()
        val json = sharedPref.getString("history", "[]")
        val type = object : TypeToken<List<HistoryItem>>() {}.type
        historyItems.clear()
        historyItems.addAll(gson.fromJson(json, type))
    }

    private fun deleteHistoryItem(position: Int) {
        // Remove from the list
        historyItems.removeAt(position)
        // Notify adapter
        adapter.notifyItemRemoved(position)
        // Update SharedPreferences
        val sharedPref = requireContext().getSharedPreferences("PlantHistory", Context.MODE_PRIVATE)
        val gson = Gson()
        sharedPref.edit()
            .putString("history", gson.toJson(historyItems))
            .apply()
    }

    private fun showPlantDetails(item: HistoryItem) {
        val plant = plantsList.firstOrNull { it.name.equals(item.plantName, ignoreCase = true) } ?: return

        val bottomSheet = BottomSheetDialog(requireContext())
        val bottomSheetView = layoutInflater.inflate(R.layout.plant_details_bottom_sheet, null)
        bottomSheet.setContentView(bottomSheetView)

        // Set text fields
        bottomSheetView.findViewById<TextView>(R.id.plantName).text = plant.name
        bottomSheetView.findViewById<TextView>(R.id.scientificName).text = plant.scientific_name
        bottomSheetView.findViewById<TextView>(R.id.plantUses).text = plant.uses.joinToString(", ")
        bottomSheetView.findViewById<TextView>(R.id.plantPartsUsed).text = plant.parts_used.joinToString(", ")
        bottomSheetView.findViewById<TextView>(R.id.plantPreparation).text = plant.preparation.joinToString(", ")
        bottomSheetView.findViewById<TextView>(R.id.plantBenefits).text = plant.benefits.joinToString(", ")

        // Load user's image of the plant
        val imageView = bottomSheetView.findViewById<ImageView>(R.id.plantImage)
        if (item.imagePath.startsWith("content://")) {
            Glide.with(this).load(Uri.parse(item.imagePath)).into(imageView)
        } else {
            Glide.with(this).load(File(item.imagePath)).into(imageView)
        }

        bottomSheet.show()
    }

    companion object {
        fun addToHistory(context: Context, plant: MedicinalPlant, imagePath: String) {
            val sharedPref = context.getSharedPreferences("PlantHistory", Context.MODE_PRIVATE)
            val gson = Gson()
            val json = sharedPref.getString("history", "[]")
            val type = object : TypeToken<List<HistoryItem>>() {}.type
            val items: MutableList<HistoryItem> = gson.fromJson(json, type)

            items.add(
                HistoryItem(
                    plantName = plant.name,
                    scientificName = plant.scientific_name,
                    imagePath = imagePath
                )
            )

            sharedPref.edit()
                .putString("history", gson.toJson(items))
                .apply()
        }
    }
}
