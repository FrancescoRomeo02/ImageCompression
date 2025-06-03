# 🖼️ DCT su Immagini BMP - App Streamlit

Questa applicazione Streamlit permette di comprimere immagini BMP in scala di grigi tramite la **Trasformata Coseno Discreta (DCT-II)** applicata a blocchi.

## 🚀 Funzionalità

- Caricamento immagini BMP (solo in scala di grigi)
- Selezione dimensione del blocco `F`
- Selezione soglia di compressione `d` (frequenze alte eliminate)
- Visualizzazione dell’immagine compressa e dell’immagine originale
- Download dell’immagine compressa in formato PNG
- Visualizzazione della maschera di compressione (quali frequenze vengono mantenute o tagliate)

## 🧠 Come funziona

L'immagine viene suddivisa in blocchi `F × F`:
- Ogni blocco è trasformato tramite **DCT-II**
- I coefficienti ad alta frequenza (k + l ≥ d) vengono **azzerati**
- Il blocco viene ricostruito con la **IDCT-II**
- I blocchi vengono ricomposti per ottenere l'immagine compressa


## 📦 Requisiti
- Python 3.8+
- Librerie:
    - streamlit
    - numpy
    - Pillow
    - matplotlib
    - scipy


📄 Licenza

Questo progetto è distribuito con licenza MIT. Puoi usarlo liberamente.

---