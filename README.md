 O projektu
Tento projekt má za cíl vytvořit obrazový klasifikátor, který dokáže rozpoznávat různé druhy zvířat. Model je postaven na architektuře ResNet18 a využívá PyTorch. Cesta k úspěchu vedla přes spoustu obrázků, tunu kódu a pár nesprávně nastavených cest k souborům.

Použitá data
Obrázky jsou organizovány podle kategorií zvířat a rozdělily se do tří skupin:

Train (70 %) – Co se nenaučí tady, to už nikde.
Validation (20 %) – Pomáhá modelu ladit formu.
Test (10 %) – Zkouška ohněm na neznámých datech.

Struktura projektu

obrazkovy-editor/
├── data/
│   ├── raw/          # Nezpracovaná data
│   ├── processed/    # Augmentovaná data
│   ├── models/       # Uložené modely
│   ├── outputs/      # Logy a výstupy
├── pretrained_models/ # Předtrénované váhy
├── main_script.py     # Hlavní skript pro trénování
├── test_model.py      # Skript pro testování modelu
├── data_augmentation.py # Modul pro augmentaci dat
├── utils.py          # Pomocné funkce
├── README.md         # Tento soubor

Postup práce
Příprava dat
Ořezání nadbytečných obrázků (max. 100 na kategorii).
Rozdělení na trénovací, validační a testovací sady.
Augmentace obrázků (rotace, zrcadlení, změna jasu).

Trénování modelu
Použití ResNet18 – protože proč vynalézat kolo, když existuje předtrénovaný model.
Optimalizace pomocí CrossEntropyLoss a Adam optimizéru.
5 epoch a model měl velmi slušné výsledky.

Testování modelu
Přesnost 98,7 % – ne že by to bylo špatné, ale vždycky může být lépe.

Chyby a jejich (téměř) bezbolestná řešení
Neexistující cesty k souborům → Přidány diagnostické výpisy.
Chybějící importy → Důsledná kontrola knihoven, už žádné ModuleNotFoundError.
Testovací obrázky na špatném místě → Opraveno přemístěním a použitím os.path.exists().
Model neběžel na GPU → PyTorch si občas dělá, co chce, tak přišel na řadu výpis torch.cuda.is_available().

Spuštění projektu
Předpoklady
Python 3.10+
Knihovny PyTorch, Torchvision, PIL

 Kroky ke spuštění

Připrav data
Nezpracované obrázky umísti do data/raw/raw-img

Budoucí vylepšení
Rozšíření datasetu – Více kategorií, více obrázků, více datových chyb, co budu opravovat.
Lepší hyperparametry – Experimentování s různými vrstvami modelu.
Zrychlení inferencí – Možná menší model nebo nasazení na specializovaný hardware.