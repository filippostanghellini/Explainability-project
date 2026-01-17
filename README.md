## Explainable AI Benchmark su CUB200-2011

## 📂 Struttura 

```
project/
├── CUB_200_2011/          # Cartella del dataset (da scaricare)
├── src/
│   ├── data_utils.py      # Gestione dataset CUB200 e parsing metadati
│   ├── model.py           # Definizione architettura (ResNet50)
│   ├── explainers.py      # Wrapper per libreria Captum (IG, LIME, SHAP...) 
│   └── train.py           # Loop di training e validazione
├── exploratory.ipynb      # Notebook per esperimenti preliminari
└── README.md              
```

---

## 🛠 Moduli Implementati

### 1. Modello (`src/model.py`)

Utilizza il **Transfer Learning** partendo da una **ResNet50** pre-addestrata su **ImageNet**.

- **Modifica**: l'ultimo layer *Fully Connected* è sostituito per adattarsi alle **200 classi** di uccelli del dataset CUB.
- **Funzionalità**: supporta il **salvataggio/caricamento dei pesi** e l'**estrazione delle feature**.

---

### 2. Dati (`src/data_utils.py`)

Classe `CUBDataset` personalizzata che gestisce la complessità dei file di testo del **CUB200**:

- Incrocia `images.txt` e `image_class_labels.txt` per associare **immagini e label**.
- Gestisce le **trasformazioni** (resize, normalizzazione) necessarie per ResNet.

> **Nota**: include una funzione preliminare `get_part_annotations` per leggere le **coordinate delle parti anatomiche** (fondamentale per la fase di valutazione).

---

### 3. Explainers (`src/explainers.py`)

Un'architettura a oggetti basata su **Captum** che standardizza l'interfaccia per diversi metodi di spiegazione:

- **Gradient-based (White-box)**: Integrated Gradients, Saliency (Input Gradients).
- **Perturbation-based (Black-box)**: LIME, KernelSHAP.

**Output**: ogni explainer restituisce una **heatmap normalizzata**, pronta per la visualizzazione o la valutazione quantitativa.

---

### 4. Training (`src/train.py`)

Pipeline di **fine-tuning** veloce:

- Usa `CrossEntropyLoss` e ottimizzatore **Adam**.
- Salva automaticamente il **modello con la migliore accuratezza** sul validation set.

---

## 🚀 Come Eseguire

### Prerequisiti

```bash
pip install -r requirements.txt
```
Run exploratory notebook to test modules

---

## 📊 Stato del Progetto e Prossimi Passaggi (TODO)

Al momento il progetto è in grado di:

- [x] Caricare correttamente il dataset e le label
- [x] Addestrare il modello con buone performance
- [x] Generare spiegazioni visive (heatmap) con **4 algoritmi diversi**

### Gap Analysis – Requisiti mancanti per l'esame

Il requisito fondamentale del corso è la **valutazione quantitativa della plausibilità**.

- [ ] **Data Engineering**: aggiornare `CUBDataset` per restituire le **Ground Truth Masks** (maschere binarie create dalle coordinate delle parti anatomiche)
- [ ] **Metrica**: implementare una funzione di **Intersection over Union (IoU)** o **Energy Fraction** per misurare quanto la heatmap si sovrappone alla maschera delle parti reali
- [ ] **Benchmark**: eseguire uno script su tutto il test set per ottenere i punteggi finali (es. *"IG ha una plausibilità del 60% vs LIME 45%"*)

