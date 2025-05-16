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
        relative_key = "/".join(key.split("/")[1:])  
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

spinner_text = (f"Fetching {species} birds to compare their call to yours...")
@st.cache_resource(show_spinner=spinner_text)
def get_reducer(species: str, n_neighbors: int = 15, min_dist: float = 0.1):
    species_df = get_species_df(species)
    if species_df.empty or not any(col.startswith("dim_") for col in species_df.columns):
        st.warning(f"No embedding data for {species}.")
        return None, species_df
    embed_cols = [c for c in species_df.columns if c.startswith("dim_")]
    reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit(species_df[embed_cols].values)
    return reducer, species_df

def run_umap(reducer: UMAP, species_df: pd.DataFrame, user_emb: np.ndarray) -> pd.DataFrame:
    if reducer is None or species_df.empty or user_emb is None or user_emb.size == 0:
        return pd.DataFrame()
    embed_cols = [c for c in species_df.columns if c.startswith("dim_")]
    coords_bird = reducer.embedding_
    coords_user = reducer.transform(user_emb.reshape(1, -1))
    df2 = species_df.copy()
    df2[["umap_x", "umap_y", "umap_z"]] = coords_bird
    user_row = dict(zip(embed_cols, user_emb))
    user_row.update({"file": "You", "s3_key": "N/A-User", "umap_x": float(coords_user[0, 0]), "umap_y": float(coords_user[0, 1]), "umap_z": float(coords_user[0, 2])})
    df2 = pd.concat([df2, pd.DataFrame([user_row])], ignore_index=True)
    df2["type"] = df2["file"].apply(lambda f: "User" if f == "You" else "Bird")
    return df2

if not species_to_scrape:
    st.error("`species_to_scrape` is empty in config.py.")
    st.stop()

all_species = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]
if not all_species:
    st.error("No species available after filtering.")
    st.stop()

if "current_species" not in st.session_state:
    st.session_state.current_species = random.choice(all_species)
if "previous_species" not in st.session_state:
    st.session_state.previous_species = []
if "selected_key" not in st.session_state:
    st.session_state.selected_key = None
if "mimic_submitted" not in st.session_state:
    st.session_state.mimic_submitted = False
if "loaded_species" not in st.session_state:
    st.session_state.loaded_species = None

with st.spinner("Recording birds, please wait while we gather calls..."):
    species = st.session_state.current_species

    st.title("Are you good at making bird calls?")

    img_key = f"Images/{species}.jpg"
    try:
        img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
        st.image(img_bytes)
    except Exception:
        st.caption(f"(No image for {species})")

    s3_keys_for_species = list_audio_keys(species)
    if not s3_keys_for_species:
        st.error(f"No audio files found for {species}.")
        st.stop()

    valid_audio_keys: List[str] = []
    audio_durations: Dict[str, float] = {}
    temp_files_duration = []

    if st.session_state.loaded_species != species:
        st.session_state.loaded_species = species
        with st.spinner(f"Fetching audio details for {species}..."):
            for key in s3_keys_for_species:
                local_path = download_to_temp(key)
                temp_files_duration.append(local_path)
                try:
                    duration = get_duration(path=local_path)
                    audio_durations[key] = duration
                    if duration <= 20:
                        valid_audio_keys.append(key)
                except Exception as e:
                    st.warning(f"Could not get duration for {key}. Error: {e}")
            for f in temp_files_duration:
                Path(f).unlink(missing_ok=True)

    if not valid_audio_keys:
        valid_audio_keys = [min(audio_durations, key=audio_durations.get)] if audio_durations else s3_keys_for_species

    if not valid_audio_keys:
        st.error(f"No suitable audio for {species}.")
        st.stop()

    if st.session_state.selected_key not in valid_audio_keys or st.session_state.selected_key is None:
        st.session_state.selected_key = random.choice(valid_audio_keys)

    ref_key = st.session_state.selected_key
    ref_audio_url = presigned_url(ref_key)

    if ref_audio_url:
        st.audio(ref_audio_url)
    else:
        st.error("Could not load reference audio.")

st.divider()
st.header(f"Try to mimic the {species}!")

recorder_key = f"mimic_audio_{species}_{Path(ref_key).stem if ref_key else 'no_ref'}"
user_audio = st.audio_input("Record your attempt here:", key=recorder_key)

if user_audio and not st.session_state.mimic_submitted:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio.write(user_audio.read())
        user_audio_path = tmp_audio.name
    if Path(user_audio_path).exists() and Path(user_audio_path).stat().st_size > 0:
        relative_ref_key = "/".join(ref_key.split("/")[1:])
        if relative_ref_key in bird_embeddings:
            ref_embedding = bird_embeddings[relative_ref_key]
            user_embedding = compute_embedding(user_audio_path)
            if user_embedding.size > 0:
                similarity = cosine_similarity(ref_embedding, user_embedding)
                if similarity > 0.7:
                    score = int((similarity - 0.7) / 0.3 * 100)
                    score = max(0, min(100, score))
                else:
                    score = 0
                st.session_state.mimic_submitted = True
                st.metric("Similarity Score:", f"{score}%")
                with st.spinner("Visualizing your call..."):
                    reducer, species_df_umap = get_reducer(species)
                    if reducer and not species_df_umap.empty:
                        umap_df = run_umap(reducer, species_df_umap, user_embedding)
                        if not umap_df.empty:
                            fig = px.scatter_3d(umap_df, color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"})
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Your call is orange; real bird calls are green.")
        else:
            st.error(f"Reference embedding for {relative_ref_key} not found.")
    Path(user_audio_path).unlink(missing_ok=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("🦉 Try a new bird"):
        st.session_state.previous_species.append(species)
        st.session_state.previous_species = st.session_state.previous_species[-3:]
        candidates = [s for s in all_species if s not in st.session_state.previous_species]
        st.session_state.current_species = random.choice(candidates or all_species)
        st.session_state.selected_key = None
        st.session_state.mimic_submitted = False
        st.session_state.loaded_species = None 
        if recorder_key in st.session_state:
            st.session_state.pop(recorder_key)
        st.rerun()

with col2:
    if st.session_state.mimic_submitted:
        if st.button("🎶 Try this species again"):
            st.session_state.selected_key = random.choice(valid_audio_keys)
            st.session_state.mimic_submitted = False
            if recorder_key in st.session_state:
                st.session_state.pop(recorder_key)
            st.rerun()
    else:
        st.button("🎶 Try this species again", disabled=True)
