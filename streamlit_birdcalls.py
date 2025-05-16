"""Bird Call Mimic Game
------------------------------------------------
This Streamlit app lets users practise bird calls. For each round it:
1. Chooses a random bird species (from `config.species_to_scrape`).
2.I chose the bird species due to their funky calls, knowledge I have from guiding at Glacier National Park
2. Plays a short reference clip
3. Lets the user record their own attempt.
4. Computes Wav2Vec2 embeddings, cosine similarity (for the score) & a 3-D UMAP visualisation.
"""
import io
import random
import tempfile
from pathlib import Path
from typing import Dict, Tuple, List

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torchaudio
from librosa import get_duration
from torchaudio.functional import resample
from transformers.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2Processor
from umap import UMAP

from config import species_to_scrape

st.set_page_config(page_title="Are you good at making bird calls?", page_icon="🪶", layout="wide")

DEFAULT_BUCKET = "bird-database"

@st.cache_resource(show_spinner="Connecting to S3...")
def get_s3_client():
    try:
        session = boto3.Session(
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_REGION", "us-east-1"),
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

def list_audio_keys(species: str) -> List[str]:
    keys = []
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
            "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_sec
        )
    except Exception as e:
        st.error(f"Error generating presigned URL for {key}: {e}")
        return ""

def download_to_temp(key: str) -> str:
    suffix = Path(key).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        CLIENT.download_fileobj(S3_BUCKET, key, tmp)
        return tmp.name

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

@st.cache_resource(show_spinner="Loading Wav2Vec2 model...")
def init_model() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model(torch.zeros(1, 16000, device=device))
    return processor, model

@st.cache_data(show_spinner="Fetching pre-computed embeddings...")
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

@st.cache_data(show_spinner="Preparing species data...")
def get_species_df(species: str) -> pd.DataFrame:
    s3_audio_keys = list_audio_keys(species)
    embeddings = []
    files = []
    s3_keys = []
    temp_files = []
    for key in s3_audio_keys:
        relative_key = "/".join(key.split("/")[1:])  # Remove the "Data/" prefix
        emb = bird_embeddings.get(relative_key)
        if emb is None:
            st.warning(f"Embedding NOT found for key: {key} (relative: {relative_key})")
            local_path = download_to_temp(key)
            temp_files.append(local_path)
            emb = compute_embedding(local_path)
        if emb.size > 0:
            embeddings.append(emb)
            files.append(Path(key).name)
            s3_keys.append(key)
    for f in temp_files:
        Path(f).unlink(missing_ok=True)
    if not embeddings:
        return pd.DataFrame(columns=[f"dim_{i}" for i in range(768)] + ["file", "s3_key"])
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings[0].shape[0])])
    df["file"] = files
    df["s3_key"] = s3_keys
    return df

@st.cache_resource(show_spinner="Fitting UMAP reducer...")
def get_reducer(species: str, n_neighbors: int = 15, min_dist: float = 0.1):
    species_df = get_species_df(species)
    if species_df.empty or not any(col.startswith("dim_") for col in species_df.columns):
        st.warning(f"No embedding data for {species}.")
        return None, species_df
    embed_cols = [c for c in species_df.columns
