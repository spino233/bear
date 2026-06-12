import gradio as gr
from fastai.vision.all import *

# carico il file del modello
def is_cat(x): return x[0].isupper()
model = load_learner('export.pkl')

# Definisco una tupla contenente le due categorie predette dal modello
categories = model.dls.vocab

# map converte ogni valore del Tensor in un float
# zip associa ad ogni categories un valore estratto dalla map
# dict converte le tuple generate da zip in un dizionario  
def classify_img(img):
    pred, idx, probs = model.predict(img)
    print(pred, probs)
    return dict(zip(categories, map(float,probs)))

# Definisco gli esempi da caricare
examples = ['black.jpg', 'teddy.jpeg', 'grizzly.jpeg']

# Input Component per caricare le immagini
image = gr.Image()
# Output Component per mostrare le etichette e il livello di confidenza della classificazione
label = gr.Label()

# Crea una UI base utilizzando gradio
intf = gr.Interface(fn=classify_img, inputs=gr.Image(), outputs=gr.Label(), examples=examples)

# Lancia un web server che esegue la demo
intf.launch(inline=False)

