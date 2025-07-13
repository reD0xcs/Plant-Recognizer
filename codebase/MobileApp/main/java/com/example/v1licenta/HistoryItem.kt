package com.example.v1licenta

/**
 * Clasa de model pentru un element din istoricul de recunoastere a plantelor.
 *
 * Reprezinta o intrare in istoric care contine:
 * - Numele plantei recunoscute
 * - Numele stiintific al plantei
 * - Calea catre imaginea capturata
 * - Data si ora recunoasterii (timestamp)
 *
 * Proprietati:
 * @property plantName Numele comun al plantei
 * @property scientificName Numele stiintific al plantei
 * @property imagePath Calea catre imagine (poate fi URI content sau cale locala)
 * @property timestamp Momentul recunoasterii (default: timpul curent)
 */
data class HistoryItem(
    val plantName: String,
    val scientificName: String,
    val imagePath: String,
    val timestamp: Long = System.currentTimeMillis()
)
