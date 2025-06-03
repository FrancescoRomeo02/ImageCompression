# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.fft import dctn, idctn

# Configura la pagina
st.set_page_config(page_title="DCT su Immagini BMP", layout="centered")
st.title("Trasformata DCT-II su Immagini BMP")

# Caricamento file
uploaded_file = st.file_uploader("Carica un file BMP", type=["bmp"])

if uploaded_file:
    # Converti l'immagine in scala di grigi
    original_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(original_bytes)).convert("L")
    arr = np.array(img)
    H, W = arr.shape
    st.write(f"Dimensione immagine: {W} × {H} px")

    # Parametri utente
    max_F = min(H, W)
    F = st.number_input("Dimensione del blocco F", min_value=1, max_value=max_F, value=10)
    max_d = 2 * F - 2
    d = st.slider("Soglia di taglio delle frequenze d", 0, max_d, min(10, max_d))

    # Ritaglia immagine
    H2, W2 = (H // F) * F, (W // F) * F
    arr_crop = arr[:H2, :W2]
    out = np.zeros_like(arr_crop)

    # Crea maschera DCT
    k, l = np.arange(F)[:, None], np.arange(F)[None, :]
    mask = (k + l) >= d

    # Applica DCT per blocchi
    for bi in range(0, H2, F):
        for bj in range(0, W2, F):
            block = arr_crop[bi:bi+F, bj:bj+F].astype(float)
            C = dctn(block, norm='ortho')
            C[mask] = 0.0
            rec = np.rint(idctn(C, norm='ortho')).astype(int)
            rec = np.clip(rec, 0, 255)
            out[bi:bi+F, bj:bj+F] = rec

    # Salva immagine ricostruita
    reconstructed_img = Image.fromarray(out.astype(np.uint8))
    buf = io.BytesIO()
    reconstructed_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Visualizza immagini
    col1, col2 = st.columns(2)
    with col1:
        st.image(arr_crop, caption="Immagine originale", clamp=True, channels="L")
        st.caption(f"Dimensione file originale: {len(original_bytes)//1024} KB")
    with col2:
        st.image(out, caption=f"Immagine compressa (F={F}, d={d})", clamp=True, channels="L")
        st.caption(f"Dimensione PNG: {len(img_bytes)//1024} KB")

    # Pulsante download
    st.download_button("📥 Scarica immagine compressa (PNG)", data=img_bytes,
                       file_name="immagine_compressa.png", mime="image/png")

    # Visualizza maschera di compressione
    c_viz = mask.astype(int)
    cmap = ListedColormap(['yellow', 'red'])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(c_viz, cmap=cmap, origin='upper', interpolation='none')
    ax.set_xticks(np.arange(F))
    ax.set_yticks(np.arange(F))
    ax.set_xticks(np.arange(-0.5, F, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, F, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('l (indice frequenza)')
    ax.set_ylabel('k (indice frequenza)')
    ax.set_title(f'Maschera compressione (k + l ≥ {d})')
    ax.legend(handles=[
        mpatches.Patch(color='red', label='Azzerato'),
        mpatches.Patch(color='yellow', label='Mantenuto')],
        loc='upper right')
    st.pyplot(fig)