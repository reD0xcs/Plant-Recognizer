package com.example.v1licenta

/**
 * Clasa de model pentru reprezentarea unei plante medicinale.
 *
 * Aceasta clasa contine toate informatiile necesare despre o planta medicinala:
 * - Denumire populara si stiintifica
 * - Utilizari traditionale
 * - Parti folosite
 * - Metode de preparare
 * - Beneficii pentru sanatate
 * - Referinta la imaginea asociata
 *
 * @property name Denumirea populara a plantei
 * @property scientific_name Denumirea stiintifica (latină) a plantei
 * @property uses Lista de utilizari traditionale
 * @property parts_used Lista cu partile plantei folosite (frunze, flori, radacini etc.)
 * @property preparation Metode de preparare (ceai, tinctura etc.)
 * @property benefits Beneficii pentru sanatate
 * @property photo_url Către imaginea plantei (URL sau cale locală)
 */
data class MedicinalPlant(
    val name: String,
    val scientific_name: String,
    val uses: List<String>,
    val parts_used: List<String>,
    val preparation: List<String>,
    val benefits: List<String>,
    val photo_url: String
)
