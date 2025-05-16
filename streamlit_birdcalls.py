"""Bird Call Mimic Game
------------------------------------------------
This Streamlit app lets users practise bird calls. For each round it:
1. Chooses a random bird species (from `config.species_to_scrape`). There are 10 options I chose from fun bird calls.
2. The reason I know bird calls is because I'm a montana master naturalist and former Glacier National Park Guide
3. Plays a short reference clip that lives in **Amazon S3** (bucket `bird-database`).
4. Lets the user record their own attempt.
5. Computes Wav2Vec2 embeddings, cosine similarity (for the score) & a 3-D UMAP visualisation.

Known issues:
*It gives everyone pretty high scores, so learning average scores from the cosine
 similarity and creating a more vague 'how good you are' score may be better

 Future Goals:
 *Add in more bird species!
 *Work on memory management so users can save their calls
 *Add fun facts about each bird so people can learn as they mimic the bird
"""

# -------------------------------------------------------------------------
# Streamlit setup (must precede any other st.* call)
# -------------------------------------------------------------------------
import streamlit as st
st.set_page_config(
    page_title="Are you good at making bird calls?",
    page_icon="🪶",
    layout="wide",
)

# -------------------------------------------------------------------------
# Imports - standard -> third-party -> local
# -------------------------------------------------------------------------
import io
import os
import random
import tempfile
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

@st.cache_resource(show_spinner="Connecting to S3...")
def get_s3_client():
    """Initialise boto3 client once per session."""
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_REGION", "us-east-1"),
    )

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
CLIENT = get_s3_client()


def list_audio_keys(species: str) -> list[str]:
    """Return all object keys for the given species folder (mp3/wav only)."""
    paginator = CLIENT.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{species}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".mp3", ".wav")):
                keys.append(key)
    return keys


def presigned_url(key: str, expires_sec: int = 3600) -> str:
    """Generate a temporary, signed URL for public playback of an S3 object."""
    return CLIENT.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires_sec,
    )


def download_to_temp(key: str) -> str:
    """Download an S3 object to a NamedTemporaryFile and return its path."""
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


@st.cache_resource(show_spinner="Loading Wav2Vec2...")
def init_model() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    _ = model(torch.zeros(1, 16000, device=device))  # warm-up
    return processor, model


@st.cache_data(show_spinner="Fetching embeddings...")
def load_all_embeddings() -> Dict[str, np.ndarray]:
    """Load the single embeddings.npz stored in S3 -> dict[clip_key] = vector."""
    obj = CLIENT.get_object(Bucket=S3_BUCKET, Key="embeddings/bird_embeddings.npz")
    data = io.BytesIO(obj["Body"].read())
    npz = np.load(data)
    return {k: npz[k] for k in npz.files}


processor, model = init_model()
bird_embeddings = load_all_embeddings()

# -------------------------------------------------------------------------
# NumPy utilities
# -------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if denom == 0 else dot / denom


@torch.inference_mode()
def compute_embedding(audio_path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = resample(waveform, sr, 16000)
    waveform = waveform.to(device)

    with torch.cuda.amp.autocast():
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# -------------------------------------------------------------------------
# Data wrangling
# -------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_species_df(species: str) -> pd.DataFrame:
    keys = list_audio_keys(species)
    embs = []
    files = []
    for key in keys:
        emb = bird_embeddings.get(key)
        if emb is None:
            local_path = download_to_temp(key)
            emb = compute_embedding(local_path)
        embs.append(emb)
        files.append(Path(key).name)

    df = pd.DataFrame(embs, columns=[f"dim_{i}" for i in range(len(embs[0]))])
    df["file"] = files
    df["s3_key"] = keys
    return df


@st.cache_resource(show_spinner=False)
def get_reducer(species: str, n_neighbors: int = 15, min_dist: float = 0.1):
    species_df = get_species_df(species)
    embed_cols = [c for c in species_df.columns if c.startswith("dim_")]
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit(
        species_df[embed_cols].values
    )
    return reducer, species_df


def run_umap(reducer: UMAP, species_df: pd.DataFrame, user_emb: np.ndarray) -> pd.DataFrame:
    embed_cols = [c for c in species_df.columns if c.startswith("dim_")]
    coords_bird = reducer.embedding_
    coords_user = reducer.transform(user_emb.reshape(1, -1))

    df2 = species_df.copy()
    df2[["umap_x", "umap_y", "umap_z"]] = coords_bird

    user_row = dict(zip(embed_cols, user_emb))
    user_row.update(
        {
            "file": "You",
            "umap_x": float(coords_user[0, 0]),
            "umap_y": float(coords_user[0, 1]),
            "umap_z": float(coords_user[0, 2]),
        }
    )
    df2 = pd.concat([df2, pd.DataFrame([user_row])], ignore_index=True)
    df2["type"] = df2["file"].apply(lambda f: "User" if f == "You" else "Bird")
    return df2

# -------------------------------------------------------------------------
# Session state init
# -------------------------------------------------------------------------

all_species = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]

if "current_species" not in st.session_state:
    st.session_state.current_species = random.choice(all_species)
if "previous_species" not in st.session_state:
    st.session_state.previous_species = []
if "selected_key" not in st.session_state:
    st.session_state.selected_key = None
if "mimic_submitted" not in st.session_state:
    st.session_state.mimic_submitted = False

species = st.session_state.current_species

# ---------------------------------------------------------------------
# UI - reference image and audio
# ---------------------------------------------------------------------
st.title("Are you good at making bird calls?")

img_key = f"Images/{species}.jpg"
try:
    img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    st.image(img_bytes)
except Exception:
    st.write("(no image available for this species)")

# Choose a reference clip
keys = list_audio_keys(species)
valid_keys: List[str] = []
durations: Dict[str, float] = {}
for key in keys:
    try:
        tmp = download_to_temp(key)
        d = get_duration(path=tmp)
        durations[key] = d
        if d <= 20:
            valid_keys.append(key)
    except Exception:
        continue

if not valid_keys:
    # fall back to the shortest clip if none are <= 20 s
    if durations:
        valid_keys = [min(durations, key=durations.get)]
    else:
        valid_keys = keys

if st.session_state.selected_key not in valid_keys:
    st.session_state.selected_key = random.choice(valid_keys)

ref_key = st.session_state.selected_key
st.audio(presigned_url(ref_key), format="audio/mpeg")

# ---------------------------------------------------------------------
# User recording, similarity score and UMAP
# ---------------------------------------------------------------------
st.divider()
st.header(f"Try to mimic the {species}!")

recorder_key = f"mimic_audio_{species}_{Path(ref_key).stem}"
mimic = st.audio_input("Record your attempt here", key=recorder_key)

if mimic and not st.session_state.mimic_submitted:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(mimic.read())
        user_path = tmp.name

    ref_emb = bird_embeddings[ref_key]
    user_emb = compute_embedding(user_path)
    sim = cosine_similarity(ref_emb, user_emb)
    score = int((sim + 1) / 2 * 100)

    st.session_state.mimic_submitted = True

    st.metric("Similarity", f"{score}%")

    with st.spinner("Plotting your call among the bird's calls ..."):
        reducer, species_df = get_reducer(species)
        df_plot = run_umap(reducer, species_df, user_emb=user_emb)
        fig = px.scatter_3d(
            df_plot,
            color_discrete_map={"Bird": "#	babd8d", "User": "#fa9500"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Your call is orange; real bird calls are green.")

# ---------------------------------------------------------------------
# Navigation buttons to try again or do a new bird
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🦉 Try a new bird"):
        st.session_state.previous_species.append(species)
        st.session_state.previous_species = st.session_state.previous_species[-3:]

        candidates = [s for s in all_species if s not in st.session_state.previous_species]
        st.session_state.current_species = random.choice(candidates or all_species)

        st.session_state.selected_key = None
        st.session_state.mimic_submitted = False
        st.session_state.pop(recorder_key, None)
        st.rerun()

with col2:
    if st.session_state.mimic_submitted:
        if st.button("🎶 Try the call again"):
            st.session_state.selected_key = random.choice(valid_keys)
            st.session_state.mimic_submitted = False
            st.session_state.pop(recorder_key, None)
            st.rerun()
    else:
        st.button("🎶 Try the call again", disabled=True)

