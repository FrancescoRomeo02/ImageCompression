import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy.fft import dctn, idctn

# Configurazione della pagina Streamlit
st.set_page_config(page_title="DCT su Immagini BMP", layout="centered")
st.title("Trasformata DCT-II su Immagini BMP")

# Caricamento immagine BMP
uploaded_file = st.file_uploader("Carica un file BMP", type=["bmp"])

if uploaded_file:
    original_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(original_bytes)).convert("L")
    arr = np.array(img)
    H, W = arr.shape
    st.write(f"Dimensione immagine: {W} × {H} px")

    max_F = min(H, W)

    # Parametri di compressione
    F = st.number_input("Dimensione del blocco F",
                        min_value=1, max_value=max_F, value=10, step=1)
    max_d = 2 * F - 2
    d = st.slider(f"Soglia di taglio delle frequenze d (0 ≤ d ≤ {max_d})",
                  min_value=0, max_value=max_d, value=min(10, max_d))

    # Ritaglio dell'immagine per essere divisibile per F
    H2, W2 = (H // F) * F, (W // F) * F
    arr_crop = arr[:H2, :W2]
    out = np.zeros_like(arr_crop)

    # Maschera compressione
    k, l = np.arange(F)[:, None], np.arange(F)[None, :]
    mask = (k + l) >= d

    # Applicazione DCT blocco per blocco
    for bi in range(0, H2, F):
        for bj in range(0, W2, F):
            block = arr_crop[bi:bi+F, bj:bj+F].astype(float)
            C = dctn(block, norm='ortho')
            C[mask] = 0.0
            rec = np.rint(idctn(C, norm='ortho')).astype(int)
            rec = np.clip(rec, 0, 255)
            out[bi:bi+F, bj:bj+F] = rec

    # Conversione per salvataggio in PNG
    reconstructed_img = Image.fromarray(out.astype(np.uint8))
    buf = io.BytesIO()
    reconstructed_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Visualizzazione immagini con dimensioni file
    col1, col2 = st.columns(2)

    with col1:
        st.image(arr_crop, caption="Immagine originale", clamp=True, channels="L")
        st.caption(f"Dimensione file originale: {len(original_bytes)//1024} KB")

    with col2:
        st.image(out, caption=f"Immagine ricostruita (F = {F}, d = {d})", clamp=True, channels="L")
        st.caption(f"Dimensione file PNG: {len(img_bytes)//1024} KB")

    # Pulsante per il download in formato PNG
    st.download_button(
        label="📥 Scarica immagine compressa (PNG)",
        data=img_bytes,
        file_name="immagine_compressa.png",
        mime="image/png"
    )

    # Visualizzazione maschera
    c_viz = mask.astype(int)
    cmap = ListedColormap(['yellow', 'red'])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(c_viz, cmap=cmap, origin='upper', interpolation='none')
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