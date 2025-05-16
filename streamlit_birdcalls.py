# Bird Call Mimic Game  – DEBUG VERSION
# ------------------------------------------------
# Changes vs original
#   • Added explicit seek(0) + empty‑recording guard after st.audio_input
#   • Added "<=10 s" hint + max_duration param to st.audio_input
#   • Normalise stereo→mono and sample‑rate in compute_embedding
#   • Fallback lookup for pre‑computed embeddings by base filename
#   • Scattered st.write / st.error for easier live debugging
# ------------------------------------------------

import streamlit as st
st.set_page_config(
    page_title="Are you good at making bird calls? (debug)",
    page_icon="🪶",
    layout="wide",
)

import io
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torchaudio
from librosa import get_duration
from torchaudio.functional import resample
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from umap import UMAP

from config import species_to_scrape

# -------------------------------------------------------------------------
# AWS / S3 helpers
# -------------------------------------------------------------------------
DEFAULT_BUCKET = "bird-database"

@st.cache_resource(show_spinner="Connecting to S3…")
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_REGION", "us-east-1"),
    )

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
CLIENT = get_s3_client()


def list_audio_files(species: str) -> list[str]:
    paginator = CLIENT.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"Data/{species}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".mp3", ".wav")):
                keys.append(key)
    return keys


def presigned_url(key: str, expires_sec: int = 3600) -> str:
    return CLIENT.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires_sec,
    )


def download_to_temp(key: str) -> str:
    suffix = Path(key).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        CLIENT.download_fileobj(S3_BUCKET, key, tmp)
        return tmp.name

# -------------------------------------------------------------------------
# Wav2Vec2 model & pre-computed embeddings
# -------------------------------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


@st.cache_resource
def init_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    return processor, model


@st.cache_data
def load_all_embeddings():
    obj = CLIENT.get_object(Bucket=S3_BUCKET, Key="all_embeddings.pt")
    buf = io.BytesIO(obj["Body"].read())
    embeddings = torch.load(buf, map_location="cpu")
    # 🔑 normalise dictionary keys to base filename only
    return {Path(k).name: v.numpy() for k, v in embeddings.items()}


processor, model = init_model()
bird_embeddings: dict[str, np.ndarray] = load_all_embeddings()

# -------------------------------------------------------------------------
# NumPy utilities
# -------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm_vec1 = np.linalg.norm(v1)
    norm_vec2 = np.linalg.norm(v2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot / (norm_vec1 * norm_vec2)


@torch.inference_mode()
def compute_embedding(audio_path: str) -> np.ndarray:
    """Load audio, down‑mix to mono, resample to 16 kHz, return Wav2Vec2 embedding."""
    waveform, sr = torchaudio.load(audio_path)

    # stereo → mono (average) ------------------------------------------------
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # resample --------------------------------------------------------------
    if sr != 16_000:
        waveform = resample(waveform, sr, 16_000)

    waveform = waveform.to(device)

    with torch.cuda.amp.autocast(dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
        inputs = processor(waveform.squeeze(), sampling_rate=16_000, return_tensors="pt").to(device)
        outputs = model(**inputs)

    # mean‑pool -------------------------------------------------------------
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# -------------------------------------------------------------------------
# Data wrangling helpers
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_species_df(species: str) -> pd.DataFrame:
    keys = list_audio_files(species)
    embs = []
    files = []

    for key in keys:
        emb = bird_embeddings.get(Path(key).name)  # lookup by filename only
        if emb is None:
            local_path = download_to_temp(key)
            emb = compute_embedding(local_path)
        embs.append(emb)
        files.append(Path(key).name)

    df = pd.DataFrame(embs, columns=[f"dim_{i}" for i in range(len(embs[0]))])
    df["file"] = files
    df["s3_key"] = keys
    return df


def run_umap(df: pd.DataFrame, user_emb: np.ndarray,
             n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
    embed_cols = [c for c in df.columns if c.startswith("dim_")]
    df_user = pd.DataFrame(user_emb.reshape(1, -1), columns=embed_cols)
    df_user["file"] = ["You"]
    df2 = pd.concat([df, df_user], ignore_index=True)

    reducer = UMAP(n_components=3, n_neighbors=n_neighbors,
                   min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(df2[embed_cols].values)
    df2[["umap_x", "umap_y", "umap_z"]] = coords
    df2["type"] = df2["file"].apply(lambda f: "User" if f == "You" else "Bird")
    return df2

# -------------------------------------------------------------------------
# App state initialisation
# -------------------------------------------------------------------------
with st.spinner("Fetching birds and recording calls in the field – please wait …"):
    all_species = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]

    ss = st.session_state
    if "current_species" not in ss:
        ss.current_species = random.choice(all_species)
    if "previous_species" not in ss:
        ss.previous_species = []
    if "selected_key" not in ss:
        ss.selected_key = None
    if "mimic_submitted" not in ss:
        ss.mimic_submitted = False

    species = ss.current_species

# -------------------------------------------------------------------------
# UI – reference image + audio
# -------------------------------------------------------------------------
st.title("Are you good at making bird calls? (debug)")

img_key = f"Images/{species}.jpg"
try:
    img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    st.image(img_bytes)
except Exception:
    st.write("(no image available for this species)")

# Choose reference clip -----------------------------------------------------
audio_files = list_audio_files(species)
durations: dict[str, float] = {}
valid_files: list[str] = []
for key in audio_files:
    try:
        tmp = download_to_temp(key)
        d = get_duration(path=tmp)
        durations[key] = d
        if d <= 20:
            valid_files.append(key)
    except Exception as e:
        st.write(f"⚠️ Could not load {key}: {e}")

if not valid_files:
    valid_files = [min(durations, key=durations.get)] if durations else audio_files

if ss.selected_key not in valid_files:
    ss.selected_key = random.choice(valid_files)

ref_key = ss.selected_key
audio_url = presigned_url(ref_key)
st.audio(audio_url, format="audio/mpeg")

# -------------------------------------------------------------------------
# User recording, similarity score and UMAP
# -------------------------------------------------------------------------
st.divider()
st.header(f"Try to mimic the {species}!")

recorder_key = f"mimic_audio_{species}_{Path(ref_key).stem}"
# 🎤  Added max_duration=10 for stability
mimic = st.audio_input("Record your attempt here (≤10 s)", key=recorder_key, max_duration=10)

if mimic and not ss.mimic_submitted:
    mimic.seek(0)                           # 🔑 reset pointer!
    raw_bytes = mimic.read()

    if len(raw_bytes) == 0:
        st.error("⚠️ The recorded file is empty. This usually means the browser failed to capture audio.\nTry a shorter clip or a different browser.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_bytes)
        user_path = tmp.name

    # ------------- get reference embedding --------------------------------
    species_emb = bird_embeddings.get(Path(ref_key).name)
    if species_emb is None:
        st.write("Embedding not pre‑computed; computing now …")
        ref_local = download_to_temp(ref_key)
        species_emb = compute_embedding(ref_local)

    user_emb = compute_embedding(user_path)
    sim = cosine_similarity(species_emb, user_emb)
    score = int((sim + 1) / 2 * 100)

    ss.mimic_submitted = True

    st.divider()
    st.metric("Your call was this similar:", f"{score}%")

    st.divider()
    progress_text = "Comparing your call to the bird database …"
    my_bar = st.progress(0, text=progress_text)
    for pc in range(120):
        time.sleep(0.01)
        my_bar.progress(pc + 1, text=progress_text)

    df = get_species_df(species)
    with st.spinner("Computing UMAP …"):
        df_umap = run_umap(df, user_emb=user_emb)
        fig = px.scatter_3d(
            df_umap,
            x="umap_x", y="umap_y", z="umap_z",
            color="type", hover_name="file",
            color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Your call (orange) vs bird calls (olive).")

# -------------------------------------------------------------------------
# Navigation buttons --------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🦉 Try a new bird"):
        ss.previous_species.append(species)
        if len(ss.previous_species) > 3:
            ss.previous_species.pop(0)
        candidates = [s for s in all_species if s != species and s not in ss.previous_species]
        if not candidates:
            ss.previous_species = []
            candidates = [s for s in all_species if s != species]
        ss.current_species = random.choice(candidates)
        ss.selected_key = None
        ss.mimic_submitted = False
        ss.pop(recorder_key, None)
        st.rerun()

with col2:
    if ss.mimic_submitted:
        if st.button("🎶 Try the call again"):
            ss.selected_key = random.choice(valid_files)
            ss.mimic_submitted = False
            ss.pop(recorder_key, None)
            st.rerun()
    else:
        st.button("🎶 Try the call again", disabled=True)
