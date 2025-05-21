"""Bird Call Mimic Game
------------------------------------------------
This Streamlit app lets users practise bird calls. For each round it:
1. Chooses a random bird species from 10 options
2. I chose the bird species due to their funky calls, knowledge I have from guiding at Glacier National Park
3. Plays a short reference clip
4. Lets the user record their own attempt.
5. Computes Wav2Vec2 embeddings and cosine similarity (for the score).
"""

import io
import json
import random
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

import boto3
import numpy as np
import streamlit as st
import torch
import torchaudio
from torchaudio.functional import resample
from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Processor

from config import species_to_scrape

# Page config
st.set_page_config(
    page_title="Are you good at making bird calls?",
    page_icon="🐔",
    layout="wide",
)

DEFAULT_BUCKET = "bird-database"

@st.cache_resource(show_spinner="Connecting to S3...")
def get_s3_client():
    try:
        session = boto3.Session(
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-2",
        )
        return session.client(
            "s3",
            config=boto3.session.Config(signature_version="s3v4"),
        )
    except Exception as e:
        st.error(f"S3 Connection Error: {e}")
        st.stop()

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
CLIENT = get_s3_client()

@st.cache_data(show_spinner="Loading durations file...")
def load_audio_durations() -> Dict[str, float]:
    durations_key = "audio_durations.json"  # Adjust if you uploaded it elsewhere
    try:
        obj = CLIENT.get_object(Bucket=S3_BUCKET, Key=durations_key)
        return json.load(obj["Body"])
    except Exception as e:
        st.error(f"Error loading durations: {e}")
        return {}

@st.cache_data(show_spinner="Listing S3 audio keys...")
def list_audio_keys(species: str) -> List[str]:
    keys: List[str] = []
    try:
        paginator = CLIENT.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"Data/{species}/"):
            for obj in page.get("Contents", []):
                if obj["Key"].lower().endswith((".mp3", ".wav")):
                    keys.append(obj["Key"])
    except Exception as e:
        st.error(f"Error listing audio keys for {species}: {e}")
    return keys

def presigned_url(key: str, expires_sec: int = 3600) -> str:
    try:
        return CLIENT.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires_sec,
        )
    except Exception as e:
        st.error(f"Error generating presigned URL for {key}: {e}")
        return ""

def download_audio(key: str) -> str:
    suffix = Path(key).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        CLIENT.download_fileobj(S3_BUCKET, key, tmp)
        return tmp.name

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

@st.cache_resource(show_spinner="Running from angry eagles...")
def init_model() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model(torch.zeros(1, 16000, device=device))
    return processor, model

@st.cache_data(show_spinner="Fetching hummingbird beed...")
def load_all_embeddings() -> Dict[str, np.ndarray]:
    embeddings_key = "all_embeddings.pt"
    try:
        obj = CLIENT.get_object(Bucket=S3_BUCKET, Key=embeddings_key)
        buf = io.BytesIO(obj["Body"].read())
        embeddings = torch.load(buf, map_location="cpu")
        return {k: v.cpu().numpy() for k, v in embeddings.items()}
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return {}

processor, model = init_model()
bird_embeddings = load_all_embeddings()
audio_durations = load_audio_durations()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / denom if denom != 0 else 0.0

@torch.inference_mode()
def compute_embedding(audio_path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = resample(waveform, sr, 16000)
    waveform = waveform.to(device)
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

if not species_to_scrape:
    st.error("`species_to_scrape` is empty in config.py.")
    st.stop()

all_species = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]
if not all_species:
    st.error("No species available after filtering.")
    st.stop()

# Initialize session state
for key, val in {
    "current_species": random.choice(all_species),
    "previous_species": [],
    "selected_key": None,
    "mimic_submitted": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Load species audio keys
species = st.session_state.current_species
s3_keys_for_species = list_audio_keys(species)
if not s3_keys_for_species:
    st.error(f"No audio files found for {species}.")
    st.stop()

# Filter audio keys by duration
valid_audio_keys = [k for k in s3_keys_for_species if audio_durations.get(k, 999) <= 20]
if not valid_audio_keys:
    st.error(f"No suitable short audio found for {species}.")
    st.stop()

# Choose or keep selected reference audio
if not st.session_state.selected_key or st.session_state.selected_key not in valid_audio_keys:
    st.session_state.selected_key = random.choice(valid_audio_keys)
ref_key = st.session_state.selected_key

# Show title and image
st.title("Are you good at making bird calls?")
img_key = f"Images/{species}.jpg"
try:
    img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    st.image(img_bytes)
except Exception:
    st.caption(f"(No image for {species})")

# Play reference audio
ref_url = presigned_url(ref_key)
if ref_url:
    st.audio(ref_url)
else:
    st.error("Could not load reference audio.")

st.divider()
st.header(f"Try to mimic the {species}!")

# Audio input
recorder_key = f"mimic_audio_{species}_{Path(ref_key).stem}"
user_audio = st.audio_input("Record your attempt here:", key=recorder_key)

if user_audio and not st.session_state.mimic_submitted:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(user_audio.read())
        user_path = tmp.name
    if Path(user_path).stat().st_size > 0:
        rel_key = "/".join(ref_key.split("/")[1:])
        with st.spinner("Analyzing your recording..."):
            if rel_key in bird_embeddings:
                ref_emb = bird_embeddings[rel_key]
                usr_emb = compute_embedding(user_path)
                sim = cosine_similarity(ref_emb, usr_emb)
                score = int((sim - 0.7)/0.3*100) if sim > 0.7 else 0
                score = max(0, min(100, score))
                st.session_state.mimic_submitted = True
                st.metric("Similarity Score:", f"{score}%")
            else:
                st.error(f"Reference embedding for {rel_key} not found in loaded embeddings.")
        Path(user_path).unlink(missing_ok=True)

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🦉 Try a new bird"):
        st.session_state.previous_species.append(species)
        st.session_state.previous_species = st.session_state.previous_species[-3:]
        cand = [s for s in all_species if s not in st.session_state.previous_species]
        st.session_state.current_species = random.choice(cand or all_species)
        st.session_state.selected_key = None
        st.session_state.mimic_submitted = False
        if recorder_key in st.session_state:
            del st.session_state[recorder_key]
        st.rerun()
with col2:
    btn = st.button("🎶 Try this species again", disabled=not st.session_state.mimic_submitted)
    if btn:
        st.session_state.selected_key = random.choice(valid_audio_keys)
        st.session_state.mimic_submitted = False
        if recorder_key in st.session_state:
            del st.session_state[recorder_key]
        st.rerun()
