# Bird Call Mimic Game  – VERBOSE DEBUG VERSION WITH CUSTOM SPLASH SPINNER
# =====================================================================
# This single‑file app contains:
#   • Page‑level splash spinner (CSS/JS) shown immediately on first paint.
#   • Toggleable sidebar **Debug mode** that streams a step‑by‑step log.
#   • Robust audio handling (seek(0), empty‑bytes guard, stereo→mono, etc.).
#   • Fallback embedding lookup by base filename.
#   • ZERO Streamlit‑beta arguments — compatible with the public “Community”
#     hosting tier (Streamlit Cloud) which currently runs v1.32.*.
# ---------------------------------------------------------------------

import io
import random
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torchaudio
from librosa import get_duration
from torchaudio.functional import resample
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from umap import UMAP

from config import species_to_scrape

# ──────────────────────────────────────────────────────────────────────
# 💫  Page & splash spinner
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Are you good at making bird calls? (debug)",
    page_icon="🪶",
    layout="wide",
)

# full‑viewport overlay with CSS spinner – removed once `window.load` fires
st.markdown(
    """
    <style>
        /* full‑screen cover */
        #init-spinner {position: fixed; inset: 0; display:flex; align-items:center; justify-content:center;
                       background: var(--background-color); z-index: 99999;}
        .loader {border: 6px solid #f3f3f3; border-top: 6px solid var(--primary-color);
                  border-radius: 50%; width: 64px; height: 64px;
                  animation: spin 0.8s linear infinite;}
        @keyframes spin {0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
    </style>
    <div id="init-spinner"><div class="loader"></div></div>
    <script>window.addEventListener('load',()=>{const s=document.getElementById('init-spinner'); if(s) s.style.display='none';});</script>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────
# 🔧  Debug helper
# ──────────────────────────────────────────────────────────────────────

DEBUG = st.sidebar.toggle("🔧 Debug mode", value=True)
_log: List[str] = []

def dbg(msg: str):
    if DEBUG:
        _log.append(msg)
        st.sidebar.write(msg)

# ──────────────────────────────────────────────────────────────────────
# AWS / S3 utility
# ──────────────────────────────────────────────────────────────────────

DEFAULT_BUCKET = "bird-database"

@st.cache_resource(show_spinner="Connecting to S3 …")
def get_s3_client():
    dbg("initialising S3 client")
    return boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets.get("AWS_REGION", "us-east-1"),
    )

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
CLIENT = get_s3_client()


def list_audio_files(species: str) -> List[str]:
    paginator = CLIENT.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"Data/{species}/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith((".wav", ".mp3")):
                keys.append(k)
    dbg(f"{species}: {len(keys)} audio files found")
    return keys


def presigned_url(key: str, *, ttl: int = 3600) -> str:
    return CLIENT.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=ttl
    )


def download_to_temp(key: str) -> str:
    dbg(f"download→temp: {key}")
    suffix = Path(key).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        CLIENT.download_fileobj(S3_BUCKET, key, tmp)
        return tmp.name

# ──────────────────────────────────────────────────────────────────────
# Model + embeddings
# ──────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner="Loading Wav2Vec2 model …")
def init_model():
    dbg("loading Wav2Vec2")
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    mdl = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    return proc, mdl

processor, model = init_model()

@st.cache_data(show_spinner="Loading pre‑computed embeddings …")
def load_all_embeddings() -> Dict[str, np.ndarray]:
    obj = CLIENT.get_object(Bucket=S3_BUCKET, Key="all_embeddings.pt")
    buf = io.BytesIO(obj["Body"].read())
    raw: Dict[str, torch.Tensor] = torch.load(buf, map_location="cpu")
    dbg(f"embeddings loaded: {len(raw)} keys")
    return {Path(k).name: v.numpy() for k, v in raw.items()}

bird_embeddings = load_all_embeddings()

# ──────────────────────────────────────────────────────────────────────
# Audio → embedding utilities
# ──────────────────────────────────────────────────────────────────────

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if denom == 0 else float(np.dot(v1, v2) / denom)

@torch.inference_mode()
def compute_embedding(wav_path: str) -> np.ndarray:
    dbg(f"embedding: {wav_path}")
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo→mono
    if sr != 16_000:
        wav = resample(wav, sr, 16_000)
    wav = wav.to(DEVICE)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        out = model(**processor(wav.squeeze(), sampling_rate=16_000, return_tensors="pt").to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_species_df(species: str) -> pd.DataFrame:
    keys = list_audio_files(species)
    embs, files = [], []
    for k in keys:
        emb = bird_embeddings.get(Path(k).name)
        if emb is None:
            emb = compute_embedding(download_to_temp(k))
        embs.append(emb)
        files.append(Path(k).name)
    cols = [f"dim_{i}" for i in range(len(embs[0]))]
    df = pd.DataFrame(embs, columns=cols)
    df["file"], df["s3_key"] = files, keys
    return df


def run_umap(df: pd.DataFrame, user_emb: np.ndarray) -> pd.DataFrame:
    dbg("running UMAP")
    embed_cols = [c for c in df.columns if c.startswith("dim_")]
    df_user = pd.DataFrame(user_emb.reshape(1, -1), columns=embed_cols)
    df_user["file"] = ["You"]
    df2 = pd.concat([df, df_user], ignore_index=True)
    reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    xyz = reducer.fit_transform(df2[embed_cols].values)
    df2[["umap_x", "umap_y", "umap_z"]] = xyz
    df2["type"] = np.where(df2["file"] == "You", "User", "Bird")
    return df2

# ──────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────

ess = st.session_state
ALL_SPECIES = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]
if "current_species" not in ess:
    ess.current_species = random.choice(ALL_SPECIES)
if "previous_species" not in ess:
    ess.previous_species = []
if "selected_key" not in ess:
    ess.selected_key = None
if "mimic_submitted" not in ess:
    ess.mimic_submitted = False

species = ess.current_species

# ──────────────────────────────────────────────────────────────────────
# UI – reference image & audio
# ──────────────────────────────────────────────────────────────────────

st.title("Are you good at making bird calls? (debug)")

# image
img_key = f"Images/{species}.jpg"
try:
    img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    st.image(img_bytes)
except Exception as e:
    dbg(f"img fail: {e}")
    st.write("(no image available for this species)")

# reference clip selection
all_keys = list_audio_files(species)
valid_keys, durations = [], {}
for k in all_keys:
    try:
        d = get_duration(path=download_to_temp(k))
        durations[k] = d
        if d <= 20:
            valid_keys.append(k)
    except Exception as e:
        dbg(f"duration check failed for {k}: {e}")

if not valid_keys:
    valid_keys = [min(durations, key=durations.get)] if durations else all_keys

if ess.selected_key not in valid_keys:
    ess.selected_key = random.choice(valid_keys)

ref_key = ess.selected_key
st.audio(presigned_url(ref_key), format="audio/mpeg")

dbg(f"reference clip → {ref_key}")

# ──────────────────────────────────────────────────────────────────────
# Recording & similarity
# ──────────────────────────────────────────────────────────────────────

st.divider()
st.header(f"Try to mimic the {species}!")

rec_key = f"mimic_audio_{species}_{Path(ref_key).stem}"
# Recording widget (no unsupported arguments)
rec_key = f"mimic_audio_{species}_{Path(ref_key).stem}"
mimic = st.audio_input("Record your attempt here", key=rec_key)

if mimic and not ess.mimic_submitted:
    # Reset buffer and guard against empty recordings ---------------------------------
    mimic.seek(0)
    raw_bytes = mimic.read()
    dbg(f"recording size: {len(raw_bytes)} bytes")
    if len(raw_bytes) == 0:
        st.error("⚠️ The recorded file is empty — browser recording failed. Try a shorter clip or a different browser.")
        st.stop()

    # Write recording to a temp WAV file ---------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw_bytes)
        user_path = tmp.name

    # Get / compute reference embedding ---------------------------------------------
    species_emb = bird_embeddings.get(Path(ref_key).name)
    if species_emb is None:
        dbg("reference embedding missing; computing now …")
        species_emb = compute_embedding(download_to_temp(ref_key))

    # Compute user embedding ---------------------------------------------------------
    user_emb = compute_embedding(user_path)

    # Cosine similarity & score ------------------------------------------------------
    sim = cosine_similarity(species_emb, user_emb)
    score = int((sim + 1) / 2 * 100)

    ess.mimic_submitted = True

    st.divider()
    st.metric("Your call was this similar:", f"{score}%")

    # UMAP visualisation -------------------------------------------------------------
    st.divider()
    with st.spinner("Computing UMAP …"):
        df = get_species_df(species)
        df_umap = run_umap(df, user_emb)
        fig = px.scatter_3d(
            df_umap,
            x="umap_x", y="umap_y", z="umap_z",
            color="type", hover_name="file",
            color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Your call (orange) vs bird calls (olive).")

# ──────────────────────────────────────────────────────────────────────
# Navigation buttons
# ──────────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    if st.button("🦉 Try a new bird"):
        ess.previous_species.append(species)
        if len(ess.previous_species) > 3:
            ess.previous_species.pop(0)
        candidates = [s for s in ALL_SPECIES if s != species and s not in ess.previous_species]
        if not candidates:
            ess.previous_species = []
            candidates = [s for s in ALL_SPECIES if s != species]
        ess.current_species = random.choice(candidates)
        ess.selected_key = None
        ess.mimic_submitted = False
        ess.pop(rec_key, None)
        st.rerun()

with col2:
    if ess.mimic_submitted:
        if st.button("🎶 Try the call again"):
            ess.selected_key = random.choice(valid_keys)
            ess.mimic_submitted = False
            ess.pop(rec_key, None)
            st.rerun()
    else:
        st.button("🎶 Try the call again", disabled=True)

# ──────────────────────────────────────────────────────────────────────
# Footer debug log
# ──────────────────────────────────────────────────────────────────────

if DEBUG and _log:
    st.divider()
    with st.expander("🔍 Debug log"):
        for line in _log:
            st.write(line)
