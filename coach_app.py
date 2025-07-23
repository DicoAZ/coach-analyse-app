
# Dashboard uitbreiding met samenvatting en groeipercentages
from datetime import datetime
import smtplib
from email.message import EmailMessage
import numpy as np
import streamlit as st
import whisper
import spacy
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from fpdf import FPDF
from PIL import Image

import spacy.cli
spacy.cli.download("nl_core_news_sm")

# Laad Nederlands NLP-model
nlp = spacy.load("nl_core_news_sm")

# Laad Whisper-model
def load_model():
    return whisper.load_model("base")

# Transcriptie functie
def transcribe_audio(audio_file):
    model = load_model()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    result = model.transcribe(tmp_path, language="nl")
    os.remove(tmp_path)
    return result["text"]

# Analyse functie
def analyze_text(text):
    doc = nlp(text)
    total = len(list(doc.sents))
    open_vragen = [sent for sent in doc.sents if sent.text.strip().endswith("?") and any(q in sent.text.lower() for q in ["wat", "hoe", "waarom", "welke"])]
    gesloten_vragen = [sent for sent in doc.sents if sent.text.strip().endswith("?") and not any(q in sent.text.lower() for q in ["wat", "hoe", "waarom", "welke"])]
    positief = [sent for sent in doc.sents if any(word in sent.text.lower() for word in ["goed", "mooi", "knap", "prima", "netjes"])]
    negatief = [sent for sent in doc.sents if any(word in sent.text.lower() for word in ["slecht", "niet goed", "verkeerd", "onvoldoende", "teleurstellend"])]
    specifieke_feedback = [s for s in doc.sents if any(kw in s.text.lower() for kw in ["je passeerde goed", "mooie pass", "strakke voorzet"])]
    algemene_feedback = [s for s in doc.sents if any(kw in s.text.lower() for kw in ["goed gedaan", "netjes gespeeld", "prima"])]
    spelers_positief = defaultdict(int)
    spelers_negatief = defaultdict(int)
    for ent in doc.ents:
        if ent.label_ == "PER":
            context = ent.sent.text.lower()
            if any(p in context for p in ["goed", "mooi", "prima"]):
                spelers_positief[ent.text] += 1
            if any(n in context for n in ["slecht", "jammer", "niet goed"]):
                spelers_negatief[ent.text] += 1
    autonomie = [s for s in doc.sents if "wat zou jij" in s.text.lower() or "hoe denk jij" in s.text.lower()]
    verbinding = [s for s in doc.sents if "hoe voel" in s.text.lower() or "ik begrijp" in s.text.lower()]
    competentie = [s for s in doc.sents if "goed gedaan" in s.text.lower() or "je kan het" in s.text.lower()]
    return {
        "total": total,
        "open_vragen": len(open_vragen),
        "gesloten_vragen": len(gesloten_vragen),
        "positieve_uitingen": len(positief),
        "negatieve_uitingen": len(negatief),
        "specifieke_feedback": len(specifieke_feedback),
        "algemene_feedback": len(algemene_feedback),
        "autonomie": len(autonomie),
        "verbinding": len(verbinding),
        "competentie": len(competentie),
        "spelers_positief": dict(spelers_positief),
        "spelers_negatief": dict(spelers_negatief)
    }

# PDF-exportfunctie
def export_pdf(data, logo_path="AZLogo.png"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Coach Analyse", ln=True, align="C")
    pdf.image(logo_path, x=160, y=10, w=30)
    pdf.ln(20)
    pdf.set_font("Arial", size=12)
    for key, val in data.items():
        if isinstance(val, dict):
            for subkey, subval in val.items():
                pdf.cell(0, 10, f"{key} - {subkey}: {subval}", ln=True)
        else:
            pdf.cell(0, 10, f"{key}: {val}", ln=True)
    output_path = os.path.join(tempfile.gettempdir(), "coach_analyse_rapport.pdf")
    pdf.output(output_path)
    return output_path
