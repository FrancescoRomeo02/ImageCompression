import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Configurazione pagina
st.set_page_config(page_title="DCT su Immagini BMP", layout="centered")
st.title("📐 Caricamento BMP e Trasformata DCT")

uploaded_file = st.file_uploader("Carica un file BMP", type=["bmp"])

if uploaded_file:
    # Caricamento immagine e conversione in scala di grigi
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("L")
    arr = np.array(img)
    H, W = arr.shape
    st.success(f"Dimensione immagine: {W}×{H} px")

    max_F = min(H, W)

    # Input interattivo: dimensione finestra F
    F = st.number_input(
        "Scegli la dimensione del blocco F",
        min_value=1, max_value=max_F,
        step=1, value=min(8, max_F)
    )

    max_d = 2 * F - 2

    # Input interattivo: soglia frequenze d
    d = st.number_input(
        f"Scegli la soglia d (0 ≤ d ≤ {max_d})",
        min_value=0, max_value=max_d,
        step=1, value=min(8, max_d)
    )

    # Visualizza i parametri scelti
    st.write(f"Blocchi: {F}×{F}, soglia d={d}")

    # Funzioni DCT 2D
    def dct2(block): return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(coeffs): return idct(
        idct(coeffs.T, norm='ortho').T, norm='ortho')

    # Ritaglio immagine per divisibilità
    H2, W2 = (H // F) * F, (W // F) * F
    arr_crop = arr[:H2, :W2]
    out = np.zeros_like(arr_crop)

    # Maschera: True se coefficiente da azzerare
    k, l = np.arange(F)[:, None], np.arange(F)[None, :]
    c = (k + l) >= d

    for bi in range(0, H2, F):
        for bj in range(0, W2, F):
            block = arr_crop[bi:bi+F, bj:bj+F].astype(float)
            C = dct2(block)
            C[c] = 0.0
            rec = np.rint(idct2(C)).astype(int)
            rec = np.clip(rec, 0, 255)
            out[bi:bi+F, bj:bj+F] = rec

    # Visualizzazione immagini
    col1, col2 = st.columns(2)
    col1.image(arr_crop, caption="Originale", clamp=True,
               channels="L", use_column_width=True)
    col2.image(out, caption=f"Ricostruita (F={F}, d={
               d})", clamp=True, channels="L", use_column_width=True)

# Visualizzazione maschera
    c_viz = np.where((k + l) >= d, 1, 0)
    cmap = ListedColormap(['yellow', 'red'])
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(c_viz, cmap=cmap, origin='upper', interpolation='none')

# Linee della griglia tra i pixel
    ax.set_xticks(np.arange(-.5, F, 1), minor=True)
    ax.set_yticks(np.arange(-.5, F, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

# Etichette principali
    ax.set_xticks(np.arange(0, F, 1))
    ax.set_yticks(np.arange(0, F, 1))
    ax.set_xticklabels(np.arange(0, F, 1))
    ax.set_yticklabels(np.arange(0, F, 1))

    ax.set_xlabel('l (indice di frequenza)')
    ax.set_ylabel('k (indice di frequenza)')
    ax.set_title(f'Maschera DCT: soglia per k+l ≥ {d}')

    rosso = mpatches.Patch(color='red', label='azzerato')
    giallo = mpatches.Patch(color='yellow', label='mantenuto')
    ax.legend(handles=[rosso, giallo], loc='upper right')

    st.pyplot(fig)
