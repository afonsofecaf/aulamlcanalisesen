import streamlit as st
import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification

# Título e descrição da aplicação
st.title("Classificação de Sentimento com MobileBERT")
st.write("Esta aplicação classifica textos em três categorias: positivo, negativo ou neutro.")

# Carrega o tokenizer e o modelo MobileBERT com 3 rótulos
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=3)

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    return predicted_class, predictions.tolist()

# Área para entrada do texto
text_input = st.text_area("Digite o texto para análise:", "Este produto é incrível, superou todas as minhas expectativas!")

if st.button("Classificar"):
    label, probabilities = classify_text(text_input)
    st.write(f"Classe prevista: {label}")
    st.write("Probabilidades:", probabilities)
