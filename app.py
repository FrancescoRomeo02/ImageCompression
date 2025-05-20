import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from dct_utilis import scipy_dct2, idct2

# Configurazione pagina
st.set_page_config(page_title="DCT su Immagini BMP", layout="centered")
st.title("Trasformata DCT-II su Immagini BMP")

# Caricamento immagine
uploaded_file = st.file_uploader("Carica un file BMP", type=["bmp"])

if uploaded_file:
    # Carica immagine in scala di grigi
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("L")
    arr = np.array(img)
    H, W = arr.shape
    st.write(f"Dimensione immagine: {W} × {H} px")

    max_F = min(H, W)

    # Parametri di compressione visibili solo dopo upload
    F = st.number_input("Dimensione del blocco F",
                        min_value=1, max_value=max_F, value=10, step=1)
    max_d = 2 * F - 2
    d = st.slider(f"Soglia di taglio delle frequenze d (0 ≤ d ≤ {
                  max_d})", min_value=0, max_value=max_d, value=min(10, max_d))

    # Adattamento dell'immagine per essere divisibile per F
    H2, W2 = (H // F) * F, (W // F) * F
    arr_crop = arr[:H2, :W2]
    out = np.zeros_like(arr_crop)

    # Maschera per taglio delle frequenze (k + l ≥ d)
    k, l = np.arange(F)[:, None], np.arange(F)[None, :]
    mask = (k + l) >= d

    # Elaborazione per blocchi
    for bi in range(0, H2, F):
        for bj in range(0, W2, F):
            block = arr_crop[bi:bi+F, bj:bj+F].astype(float)
            C = scipy_dct2(block)
            C[mask] = 0.0
            rec = np.rint(idct2(C)).astype(int)
            rec = np.clip(rec, 0, 255)
            out[bi:bi+F, bj:bj+F] = rec

    # Visualizzazione immagini
    col1, col2 = st.columns(2)
    col1.image(arr_crop, caption="Immagine originale",
               clamp=True, channels="L")
    col2.image(out, caption=f"Immagine ricostruita (F = {
               F}, d = {d})", clamp=True, channels="L")

    # Visualizzazione della maschera
    c_viz = mask.astype(int)
    cmap = ListedColormap(['yellow', 'red'])

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(c_viz, cmap=cmap, origin='upper', interpolation='none')

    ax.set_xticks(np.arange(0, F))
    ax.set_yticks(np.arange(0, F))
    ax.set_xticklabels(np.arange(0, F))
    ax.set_yticklabels(np.arange(0, F))
    ax.set_xticks(np.arange(-0.5, F, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, F, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xlabel('l (indice di frequenza)')
    ax.set_ylabel('k (indice di frequenza)')
    ax.set_title(f'Maschera di compressione (k + l ≥ {d})')

    rosso = mpatches.Patch(color='red', label='Azzerato')
    giallo = mpatches.Patch(color='yellow', label='Mantenuto')
    ax.legend(handles=[rosso, giallo], loc='upper right')

    st.pyplot(fig)
