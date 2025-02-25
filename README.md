# 🖼️ Obrázkový Klasifikátor – Deep Learning Model

> **Klasifikace obrázků pomocí neuronové sítě na základě datasetu zvířat**  
> Projekt využívá **PyTorch**, **transformace dat**, **augmentaci obrázků** a **RandomForestClassifier**  
> **Model detekuje:** psi , koně , sloni , motýli , kočky  a další.

![GitHub last commit](https://img.shields.io/github/last-commit/Katy-Coder-Kat/Obrazkovy_klasifikator)
![GitHub issues](https://img.shields.io/github/issues/Katy-Coder-Kat/Obrazkovy_klasifikator)
![GitHub stars](https://img.shields.io/github/stars/Katy-Coder-Kat/Obrazkovy_klasifikator?style=social)

---

## 📌 **O projektu**
-  **Trénováno na datasetu zvířat**  
-  **Použitý model:** Convolutional Neural Network (CNN)  
-  **Augmentace dat** – převracení, oříznutí, změna barev  
-  **Analýza chyb** – Confusion Matrix  
-  **Vizualizace výsledků** – heatmapy a barploty  

---

## 🛠 **Použité technologie**
✅ **Python** (PyTorch, NumPy, Pandas, Matplotlib, Seaborn)  
✅ **Torchvision** (předzpracování obrázků)  
✅ **OpenCV** (práce s obrázky)  
✅ **RandomForestClassifier** (pro baseline model)  
✅ **Matplotlib + Seaborn** (vizualizace)  

---

## 📌 **Jak spustit projekt?**
### 1️⃣ **Klonování repozitáře**

git clone https://github.com/Katy-Coder-Kat/Obrazkovy_klasifikator.git
cd Obrazkovy_klasifikator

2️⃣ Instalace závislostí
pip install -r requirements.txt

3️⃣ Spuštění trénování modelu
python main_script.py

4️⃣ Testování modelu
python test_model.py

5️⃣ Vizualizace výsledků
python visualize_results.py

Ukázka výstupu
Confusion Matrix
📌 Ukazuje chyby modelu při klasifikaci obrázků.


Distribuce datasetu podle tříd
📌 Kolik obrázků obsahuje jednotlivé kategorie.


Ukázka predikce modelu
📌 Model detekoval zvíře na obrázku jako "cane" (pes).


🔥 Plán vývoje 
✅ 1. Implementace základního CNN modelu
✅ 2. Přidání augmentace dat
✅ 3. Přidání vizualizace výsledků
🟡 4. Vylepšení přesnosti modelu (fine-tuning, přidání dalších vrstev)
