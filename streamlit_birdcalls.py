"""Bird Call Mimic Game
------------------------------------------------
This Streamlit app lets users practise bird calls. For each round it:
1. Chooses a random bird species from 10 options
2. I chose the bird species due to their funky calls, knowledge I have from guiding at Glacier National Park
3. Plays a short reference clip
4. Lets the user record their own attempt.
5. Computes Wav2Vec2 embeddings, cosine similarity (for the score) & a 3-D UMAP visualisation.
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

# Prevent Streamlit's file-watcher from mis-inspecting torch internals
st.set_option('server.runOnSave', False)

st.set_page_config(page_title="Are you good at making bird calls?", page_icon="🪶", layout="wide")

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

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

@st.cache_resource(show_spinner="Running from angry eagles...")
def init_model() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    model.eval()
    # warm-up
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
    embeddings, files, s3_keys, temp_files = [], [], [], []
    for key in s3_audio_keys:
        relative = "/".join(key.split("/")[1:])
        emb = bird_embeddings.get(relative)
        if emb is None:
            st.warning(f"Embedding NOT found for key: {key}")
            local = download_to_temp(key)
            temp_files.append(local)
            emb = compute_embedding(local)
        if emb.size > 0:
            embeddings.append(emb)
            files.append(Path(key).name)
            s3_keys.append(key)
    for f in temp_files:
        Path(f).unlink(missing_ok=True)
    if not embeddings:
        cols = [f"dim_{i}" for i in range(embeddings[0].shape[0])] if embeddings else [f"dim_{i}" for i in range(768)]
        return pd.DataFrame(columns=cols + ["file", "s3_key"])
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings[0].shape[0])])
    df["file"] = files
    df["s3_key"] = s3_keys
    return df

# Remove caching on UMAP to avoid pickling issues
def get_reducer(species: str, n_neighbors: int = 15, min_dist: float = 0.1) -> Tuple[UMAP, pd.DataFrame]:
    with st.spinner("Running UMAP on embeddings…"):
        species_df = get_species_df(species)
        if species_df.empty or not any(c.startswith("dim_") for c in species_df.columns):
            st.warning(f"No embedding data for {species}.")
            return None, species_df
        cols = [c for c in species_df.columns if c.startswith("dim_")]
        reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        reducer.fit(species_df[cols].values)
    return reducer, species_df


def run_umap(reducer: UMAP, species_df: pd.DataFrame, user_emb: np.ndarray) -> pd.DataFrame:
    if reducer is None or species_df.empty or user_emb is None or user_emb.size == 0:
        return pd.DataFrame()
    cols = [c for c in species_df.columns if c.startswith("dim_")]
    coords = reducer.embedding_
    user_coord = reducer.transform(user_emb.reshape(1, -1))
    df2 = species_df.copy()
    df2[["umap_x", "umap_y", "umap_z"]] = coords
    row = dict(zip(cols, user_emb))
    row.update({"file": "You", "s3_key": "N/A-User", "umap_x": float(user_coord[0,0]), "umap_y": float(user_coord[0,1]), "umap_z": float(user_coord[0,2])})
    df2 = pd.concat([df2, pd.DataFrame([row])], ignore_index=True)
    df2["type"] = df2["file"].map(lambda f: "User" if f == "You" else "Bird")
    return df2

# Config checks
if not species_to_scrape:
    st.error("`species_to_scrape` is empty in config.py.")
    st.stop()
all_species = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]
if not all_species:
    st.error("No species available after filtering.")
    st.stop()

# Session state defaults
defaults = {
    "current_species": random.choice(all_species),
    "previous_species": [],
    "selected_key": None,
    "mimic_submitted": False,
    "loaded_species": None,
    "valid_audio_keys": None,
    "audio_durations": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
species = st.session_state.current_species

st.title("Are you good at making bird calls?")

# Species image
img_key = f"Images/{species}.jpg"
try:
    data = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    st.image(data)
except Exception:
    st.caption(f"(No image for {species})")

# Fetch species audio on first load
if st.session_state.loaded_species != species:
    with st.spinner("Gathering audio files for this species…"):
        keys = list_audio_keys(species)
        if not keys:
            st.error(f"No audio files found for {species}.")
            st.stop()
        valid, durations, tmp = [], {}, []
        for k in keys:
            p = download_to_temp(k)
            tmp.append(p)
            try:
                d = get_duration(path=p)
                durations[k] = d
                if d <= 20:
                    valid.append(k)
            except Exception as e:
                st.warning(f"Could not get duration for {k}: {e}")
        for f in tmp:
            Path(f).unlink(missing_ok=True)
        if not valid:
            valid = [min(durations, key=durations.get)] if durations else keys
        st.session_state.valid_audio_keys = valid
        st.session_state.audio_durations = durations
        st.session_state.loaded_species = species
valid_audio_keys = st.session_state.valid_audio_keys
if not valid_audio_keys:
    st.error(f"No suitable audio for {species}.")
    st.stop()

# Reference audio
if not st.session_state.selected_key or st.session_state.selected_key not in valid_audio_keys:
    st.session_state.selected_key = random.choice(valid_audio_keys)
ref_key = st.session_state.selected_key
url = presigned_url(ref_key)
if url:
    st.audio(url)
else:
    st.error("Could not load reference audio.")

st.divider()
st.header(f"Try to mimic the {species}!")

# Recording input
rec_key = f"mimic_{species}_{Path(ref_key).stem}"
user_audio = st.audio_input("Record your attempt here:", key=rec_key)
if user_audio and not st.session_state.mimic_submitted:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(user_audio.read())
        path = tmp.name
    if Path(path).exists() and Path(path).stat().st_size:
        rel = "/".join(ref_key.split("/")[1:])
        if rel in bird_embeddings:
            ref_emb = bird_embeddings[rel]
            try:
                with st.spinner("Computing your embedding…"):
                    user_emb = compute_embedding(path)
                if user_emb.size == 0:
                    st.error("Empty embedding – please try again.")
                    st.stop()
            except Exception as e:
                st.error(f"Embedding error: {e}")
                st.stop()
            sim = cosine_similarity(ref_emb, user_emb)
            score = int((sim - 0.7) / 0.3 * 100) if sim > 0.7 else 0
            score = max(0, min(100, score))
            st.session_state.mimic_submitted = True
            st.metric("Similarity Score:", f"{score}%")
            reducer, df_umap = get_reducer(species)
            if reducer and not df_umap.empty:
                df_vis = run_umap(reducer, df_umap, user_emb)
                if not df_vis.empty:
                    fig = px.scatter_3d(df_vis, x="umap_x", y="umap_y", z="umap_z",
                                         color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"})
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Your call is orange; real birds are green.")
        else:
            st.error(f"No reference embedding for {rel}.")
    Path(path).unlink(missing_ok=True)

# Navigation
col1, col2 = st.columns(2)
with col1:
    if st.button("🦉 Try a new bird"):
        st.session_state.previous_species.append(species)
        st.session_state.previous_species = st.session_state.previous_species[-3:]
        cand = [s for s in all_species if s not in st.session_state.previous_species]
        st.session_state.current_species = random.choice(cand or all_species)
        for k in ["selected_key", "mimic_submitted", "loaded_species"]:
            st.session_state[k] = None if k == "selected_key" else False
        st.rerun()
with col2:
    if st.session_state.mimic_submitted and st.button("🎶 Try again"):
        st.session_state.selected_key = random.choice(st.session_state.valid_audio_keys)
        st.session_state.mimic_submitted = False
        st.rerun()
