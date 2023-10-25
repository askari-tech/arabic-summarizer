from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import nltk
from nltk.tokenize import sent_tokenize
import requests
from fastapi.middleware.cors import CORSMiddleware

# Define the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str

class SummarizedText(BaseModel):
    text: str

# Initialize the summarization model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m_name = "marefa-nlp/summarization-arabic-english-news"

tokenizer = AutoTokenizer.from_pretrained(m_name)
model = AutoModelWithLMHead.from_pretrained(m_name).to(device)

def get_summary(text, tokenizer, model, device="cpu", num_beams=2):
    if len(text.strip()) < 50:
        return ["Please provide a longer text"]

    text = "summarize: <paragraph> " + " <paragraph> ".join([s.strip() for s in sent_tokenize(text) if s.strip() != ""]) + " </s>"
    text = text.strip().replace("\n", "")

    tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)

    summary_ids = model.generate(
        tokenized_text,
        max_length=512,
        num_beams=num_beams,
        repetition_penalty=1.5,
        length_penalty=1.0,
        early_stopping=True
    )

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return [s.strip() for s in output.split("<hl>") if s.strip() != ""]

@app.post("/summarize")
async def summarize_url(input_data: URLInput):
    # Send an HTTP GET request to the provided URL
    response = requests.get(input_data.url)

    # Extract Arabic text using regular expressions
    arabic_text = re.findall(r'[\u0600-\u06FF\s]+', response.text)  # Arabic Unicode range

    # Join the Arabic text to form a single string
    arabic_text = ' '.join(arabic_text)

    arabic_text = preprocess_text(arabic_text)

    final_resultant_text = " "

    chunk_size = 1000
    chunks = [arabic_text[i:i+chunk_size] for i in range(0, len(arabic_text), chunk_size)]

    for chunk in chunks:
        summaries = get_summary(chunk, tokenizer, model, device)
        for summary in summaries:
            final_resultant_text += summary + " "

    return SummarizedText(text=final_resultant_text)

def preprocess_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[\n\t ]+', ' ', text)
    text = text.strip()
    return text


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)