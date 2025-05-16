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

@st.cache_resource(show_spinner="Connecting to S3...")
def get_s3_client():
    st.write("Initializing S3 client...")
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_REGION", "us-east-1"),
        )
        
        # Test connection
        response = client.list_buckets()
        st.write(f"Successfully connected to S3. Found {len(response['Buckets'])} buckets.")
        for bucket in response['Buckets']:
            st.write(f"Bucket: {bucket['Name']}")
        
        return client
    except Exception as e:
        st.error(f"Error connecting to S3: {str(e)}")
        raise

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
st.write(f"Using S3 bucket: {S3_BUCKET}")
CLIENT = get_s3_client()


def list_audio_files(species: str) -> list[str]:
    st.write(f"Listing audio files for species: {species}")
    st.write(f"Looking in path: Data/{species}/")
    paginator = CLIENT.get_paginator("list_objects_v2")
    keys = []
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"Data/{species}/"):
            st.write(f"Found page with {len(page.get('Contents', []))} objects")
            for obj in page.get("Contents", []):
                key = obj["Key"]
                st.write(f"Found object: {key}")
                if key.lower().endswith((".mp3", ".wav")):
                    keys.append(key)
                    st.write(f"Added to audio keys: {key}")
        return keys
    except Exception as e:
        st.error(f"Error listing objects: {str(e)}")
        return []


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
    return {k: v.numpy() for k, v in embeddings.items()}



processor, model = init_model()
bird_embeddings = load_all_embeddings()

# -------------------------------------------------------------------------
# NumPy utilities
# -------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    norm_vec1 = np.linalg.norm(v1)
    norm_vec2 = np.linalg.norm(v2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    else:
        return dot / (norm_vec1 * norm_vec2)


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
    keys = list_audio_files(species)
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


def run_umap(df: pd.DataFrame, user_emb: np.ndarray,
             n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
    embed_cols = [c for c in df.columns if c.startswith("dim_")]
    df_user = pd.DataFrame(user_emb.reshape(1, -1), columns=embed_cols)
    df_user["file"] = ["You"]
    df = pd.concat([df, df_user], ignore_index=True)
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors,
                   min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(df[embed_cols].values)
    df2 = df.copy()
    df2["umap_x"], df2["umap_y"], df2["umap_z"] = coords[:, 0], coords[:, 1], coords[:, 2]
    df2["type"] = df2["file"].apply(lambda f: "User" if f == "You" else "Bird")
    return df2

# -------------------------------------------------------------------------
# Loading message at the start
# -------------------------------------------------------------------------
with st.spinner("Fetching birds and recording calls in the field, please wait a minute or so to load..."):
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
st.write("Debugging audio files:")
audio_files = list_audio_files(species)
st.write(f"Found {len(audio_files)} audio files for species: {species}")
if len(audio_files) > 0:
    st.write(f"First audio file path: {audio_files[0]}")

valid_files = []
durations = {}
for key in audio_files:
    try:
        tmp = download_to_temp(key)
        st.write(f"Downloaded {key} to temp file: {tmp}")
        d = get_duration(path=tmp)
        durations[key] = d
        st.write(f"Duration: {d} seconds")
        if d <= 20:
            valid_files.append(key)
            st.write(f"Added to valid files (duration <= 20s)")
        else:
            st.write(f"Not added to valid files (duration > 20s)")
    except Exception as e:
        st.write(f"Error processing {key}: {str(e)}")
        continue

st.write(f"Valid files count: {len(valid_files)}")
if not valid_files:
    st.write("No valid files found, attempting fallback...")
    if durations:
        valid_files = [min(durations, key=durations.get)]
        st.write(f"Using shortest file as fallback: {valid_files[0]}")
    else:
        valid_files = audio_files
        st.write(f"Using all files as fallback, count: {len(valid_files)}")

if st.session_state.selected_key not in valid_files:
    if valid_files:
        st.session_state.selected_key = random.choice(valid_files)
        st.write(f"Selected key: {st.session_state.selected_key}")
    else:
        st.error("No valid audio files found for this species!")

if st.session_state.selected_key:
    ref_key = st.session_state.selected_key
    signed_url = presigned_url(ref_key)
    st.write(f"Signed URL for audio: {signed_url}")
    st.audio(signed_url, format="audio/mpeg")

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

    rel_key = ref_key
    species_emb = bird_embeddings.get(rel_key)
    user_emb = compute_embedding(user_path)
    sim = cosine_similarity(species_emb, user_emb)
    score = int((sim + 1) / 2 * 100)

    st.session_state.mimic_submitted = True

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col2:
        st.metric("Your call was this similar:", f"{score}%")

    st.divider()
    progress_text = "Comparing your call to the bird database, this will take a moment..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(120):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
        
    df = get_species_df(species)
    with st.spinner("Comparing your call with the bird's calls..."):
        df_umap = run_umap(df, user_emb=user_emb)
        fig = px.scatter_3d(
            df_umap,
            x="umap_x", y="umap_y", z="umap_z",
            color="type", hover_name="file",
            color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Your call ('You') is shown in orange. Bird calls are shown in olive.")

# ---------------------------------------------------------------------
# Navigation buttons to try again or do a new bird
# ---------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🦉 Try a new bird"):
        st.session_state.previous_species.append(species)
        if len(st.session_state.previous_species) > 3:
            st.session_state.previous_species.pop(0)

        candidates = [s for s in all_species if s != species and s not in st.session_state.previous_species]
        if not candidates:
            st.session_state.previous_species = []
            candidates = [s for s in all_species if s != species]

        st.session_state.current_species = random.choice(candidates)
        st.session_state.selected_key = None
        st.session_state.mimic_submitted = False
        st.session_state.pop(recorder_key, None)
        st.rerun()

with col2:
    if st.session_state.mimic_submitted:
        if st.button("🎶 Try the call again"):
            st.session_state.selected_key = random.choice(valid_files)
            st.session_state.mimic_submitted = False
            st.session_state.pop(recorder_key, None)
            st.rerun()
    else:
        st.button("🎶 Try the call again", disabled=True)
