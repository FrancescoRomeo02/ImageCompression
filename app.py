import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dct import custom_dct2

st.set_page_config(page_title="DCT su Immagini BMP", layout="centered")
st.title("📐 Caricamento BMP e Trasformata DCT")

uploaded_file = st.file_uploader("Carica un file BMP", type=["bmp"])


def apply_dct(image: Image.Image) -> np.ndarray:
    # Converti in scala di grigi
    gray = image.convert("L")
    img_array = np.array(gray, dtype=float)

    # Applica la DCT 2D
    dct_log = custom_dct2(img_array)

    return dct_log


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Immagine originale", use_column_width=True)

    dct_image = apply_dct(image)

    # Visualizzazione DCT con matplotlib
    fig, ax = plt.subplots()
    ax.imshow(dct_image, cmap='gray')
    ax.set_title("📊 DCT (log)")
    ax.axis("off")

    st.pyplot(fig)
else:
    st.info("Carica un file BMP per visualizzare la DCT.")
