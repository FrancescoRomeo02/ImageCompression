# 🖼️ ImageCompression - DCT-II su Immagini

Questo progetto offre strumenti per la compressione di immagini tramite la **Trasformata Coseno Discreta (DCT-II)**, con implementazione personalizzata, benchmarking e una webapp interattiva basata su Streamlit.

## 📂 Struttura del progetto

- `main.py` — Menu principale per eseguire benchmark o avviare la webapp.
- `imagecompression/` — Moduli Python per DCT, benchmark, I/O e plotting.
- `scripts/run_benchmark.py` — Script per eseguire benchmark tra DCT custom e SciPy.
- `dct_streamlit_app/` — Applicazione Streamlit per la compressione di immagini BMP.
- `results/` — Output di benchmark e grafici.
- `tests/` — Test di verifica per la DCT.

## 🚀 Funzionalità principali

- **Implementazione custom della DCT-II** (1D e 2D)
- **Benchmark**: confronto prestazioni tra DCT custom e SciPy
- **Visualizzazione grafica** dei risultati di benchmark
- **Webapp Streamlit** per comprimere immagini BMP in scala di grigi, con download e visualizzazione maschera di compressione

## ▶️ Come si usa

### 1. Main Menu

Esegui il menu principale:
```bash
python main.py
```
- Scegli l’opzione "Run benchmark" per generare risultati e grafici in `results/`.
- Scegli l'opzione "Run webapp" per avviare l'applicazione Streamlit.
- Scegli l'opzione "Run tests" per eseguire i test di verifica della DCT.


## 📦 Requisiti

- Python 3.8+
- Librerie:
    - numpy
    - scipy
    - matplotlib
    - Pillow
    - streamlit

Installa i requisiti per la webapp con:
```bash
pip install -r dct_streamlit_app/requirements.txt
```

## 📜 Licenza

Questo progetto è distribuito con licenza MIT.

---
