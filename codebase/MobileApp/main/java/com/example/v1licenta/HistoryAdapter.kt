package com.example.v1licenta

import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import android.view.LayoutInflater
import com.bumptech.glide.Glide
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.net.Uri
import android.widget.ImageButton

/**
 * Adaptor pentru afisarea elementelor din istoricul de recunoastere a plantelor.
 *
 * Acest adaptor se ocupa de:
 * - Afisarea listei de elemente din istoric
 * - Managementul interactiunilor cu elementele listei
 * - Incarcarea imaginilor asociate fiecarui element
 * - Gestionarea stergerii elementelor
 *
 * Functionalitati principale:
 * - Afiseaza numele plantei si numele stiintific
 * - Afiseaza data si ora recunoasterii
 * - Incarca imaginea din storage local sau din URI
 * - Permite stergerea elementelor
 * - Gestioneaza click pe elemente pentru detalii
 *
 * Utilizare:
 * - Foloseste Glide pentru incarcarea imaginilor
 * - Accepta atat fisiere locale cat si URI-uri content
 * - Notifica modificari in lista prin sistemul RecyclerView
 */
class HistoryAdapter(
    private val items: MutableList<HistoryItem>,
    private val onClick: (HistoryItem) -> Unit,
    private val onDeleteClick: (HistoryItem, Int) -> Unit
) : RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

    /**
     * ViewHolder care mentine referintele catre view-urile unui element din lista.
     */
    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val plantName: TextView = view.findViewById(R.id.plantName)
        val scientificName: TextView = view.findViewById(R.id.scientificName)
        val plantImage: ImageView = view.findViewById(R.id.plantImage)
        val timestamp: TextView = view.findViewById(R.id.timestamp)
        val deleteButton: ImageButton = view.findViewById(R.id.deleteButton)
    }

    /**
     * Creeaza un nou ViewHolder cand este nevoie.
     */
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return ViewHolder(view)
    }

    /**
     * Lega datele unui element la view-urile din ViewHolder.
     */
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = items[position]
        holder.plantName.text = item.plantName
        holder.scientificName.text = item.scientificName
        holder.timestamp.text = SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault())
            .format(Date(item.timestamp))

        if (item.imagePath.startsWith("content://")) {
            Glide.with(holder.itemView)
                .load(Uri.parse(item.imagePath))
                .into(holder.plantImage)
        } else {
            Glide.with(holder.itemView)
                .load(File(item.imagePath))
                .into(holder.plantImage)
        }

        holder.itemView.setOnClickListener { onClick(item) }

        holder.deleteButton.setOnClickListener {
            onDeleteClick(item, position)
        }
    }

    /**
     * Returneaza numarul total de elemente din lista.
     */
    override fun getItemCount() = items.size

    /**
     * Helper pentru stergerea unui element din lista.
     * @param position Pozitia elementului de sters
     */
    fun removeAt(position: Int) {
        items.removeAt(position)
        notifyItemRemoved(position)
    }
}
