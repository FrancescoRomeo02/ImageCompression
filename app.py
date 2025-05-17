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

window_size = st.number_input("Inserisci la dimensione della finestra (F)", min_value=1,value=10)
d_thresh = st.number_input("Inserisci la soglia di taglio delle frequenze d (0 ≤ d ≤ 2F-2)", min_value=1)
if uploaded_file:
    #Caricamento e conversione dell'immagine in scala di grigi
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("L")
    arr = np.array(img)
    H, W = arr.shape

    #Asserzione F e d
    try:
        F = int(window_size)
        assert 1 <= F <= min(H, W)
    except:
        st.error(f"F deve essere un intero tra 1 e {min(H,W)}")
        st.stop()

    maxd = 2*F - 2
    try:
        d = int(d_thresh)
        assert 0 <= d <= maxd
    except:
        st.error(f"d deve essere un intero tra 0 e {maxd}")
        st.stop()

    st.write(f"Dimensione immagine: {W}×{H} px — Blocchi: {F}×{F}, soglia d={d}")

    #DCT E IDCT da scipy
    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')

    def idct2(coeffs):
        return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

    #Elaborazione a blocchi: rendere l'immagine divisibile per F
    H2, W2 = (H//F)*F, (W//F)*F
    arr_crop = arr[:H2, :W2]
    #Array vuoto per riempirlo con blocchi ricosrtruiti
    out = np.zeros_like(arr_crop)

    #Creazione maschera booleana FxF
    # La maschera è vera se la somma degli indici di frequenza è maggiore o uguale a d
    # La maschera è falsa se la somma degli indici di frequenza è minore a d
    k = np.arange(F)[:,None]
    l = np.arange(F)[None,:]
    c = (k + l) >= d

    #Scorrere tutta l'immagine a blocchi F×F: parto dal blocco in alto a sinistra e scorro a destra e poi in basso. (parto da 0,0)
    for bi in range(0, H2, F):
        for bj in range(0, W2, F):
            #Seleziono finestrella
            block = arr_crop[bi:bi+F, bj:bj+F].astype(float)
            #Calcolo la DCT2
            C = dct2(block)
            #Eliminazione frequenze alte se k+l ≥ d
            C[c] = 0.0
            #Ricostruzione del blocco partendo dalla DCT2 applicata ai blocchi rimanenti
            rec = idct2(C)
            #Approssimo all'intero pù vicino
            rec = np.rint(rec).astype(int)
            #Forzo i valori a rientrare nell'intervallo [0, 255]
            rec = np.clip(rec, 0, 255)
            #Inserisco il blocco ricostruito nell'immagine finale
            out[bi:bi+F, bj:bj+F] = rec

    # Visualizzazione delle due immagini: originale e ricostruita
    col1, col2 = st.columns(2)
    col1.image(arr_crop, caption="Originale", clamp=True, channels="L", use_column_width=True)
    col2.image(out,      caption=f"DCT2+IDCT2 F={F}, d={d}", clamp=True, channels="L", use_column_width=True)

    #Creazione della maschera DCT per la visualizzazione
    c_viz = np.zeros((F, F))
    for k in range(F):
        for l in range(F):
            if k + l >= d:
                c_viz[k, l] = 1  # 1 rappresenta coefficienti da azzerare (rosso) poichè superano la soglia
            else:
                c_viz[k, l] = 0  # 0 rappresenta coefficienti da mantenere (giallo)
    
    cmap = ListedColormap(['yellow', 'red']) 
    fig, ax = plt.subplots(figsize=(5, 5))
    
    im = ax.imshow(c_viz, cmap=cmap, origin='upper')
    
    ax.set_xticks(np.arange(-.5, F, 1), minor=True)
    ax.set_yticks(np.arange(-.5, F, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    ax.set_xticks(np.arange(0, F, 1))
    ax.set_yticks(np.arange(0, F, 1))
    ax.set_xticklabels(np.arange(0, F, 1))
    # Invertiamo l'ordine delle etichette sull'asse y per mantenere la numerazione corretta
    ax.set_yticklabels(np.arange(0, F, 1))
    
    ax.set_xlabel('l (indice di frequenza)')
    ax.set_ylabel('k (indice di frequenza)')
    ax.set_title(f'Maschera DCT: soglia per k+l ≥ {d}')
    
    rosso = mpatches.Patch(color='red', label='azzerato')
    giallo = mpatches.Patch(color='yellow', label='mantenuto')
    ax.legend(handles=[rosso,giallo], loc='upper right')
    st.pyplot(fig)