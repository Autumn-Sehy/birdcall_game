# Bird-Call Mimic Game — clean start-over version (complete)
# ==========================================================
# S3 folder layout assumed:
#   bird-database/
#       Data/<species>/<audio>.mp3 | .wav
#       Images/<species>.jpg
#       all_embeddings.pt
#
# Major fixes vs your very first draft
# ───────────────────────────────────
# • Robust S3 path handling (uses Data/<species>/… everywhere)
# • Embedding-dict keys normalised to base filename only
# • Stereo→mono + 16 kHz resample before Wav2Vec2
# • Empty-recording guard + optional max_duration on old Streamlit
# • Buttons + UMAP plot render correctly

import io, random, tempfile, time
from pathlib import Path
from typing import Dict, List, Tuple

import boto3, numpy as np, pandas as pd, plotly.express as px, streamlit as st
import torch, torchaudio
from librosa import get_duration
from torchaudio.functional import resample
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from umap import UMAP

from config import species_to_scrape   # your own list of 10 species

# ────────────────────────────────────────────────────────────────────
# Streamlit page
# ────────────────────────────────────────────────────────────────────
st.set_page_config("Are you good at making bird calls?", "🪶", layout="wide")

# ────────────────────────────────────────────────────────────────────
# S3 helpers
# ────────────────────────────────────────────────────────────────────
DEFAULT_BUCKET = "bird-database"

@st.cache_resource(show_spinner="🔌 Connecting to S3 …")
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id   = st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name         = st.secrets.get("AWS_REGION", "us-east-1"),
    )

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
CLIENT     = get_s3_client()

def list_audio_keys(species: str) -> List[str]:
    """Return S3 keys like Data/<species>/file.mp3."""
    keys = []
    paginator = CLIENT.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"Data/{species}/"):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith((".mp3", ".wav")):
                keys.append(k)
    return keys

def presigned_url(key: str, ttl: int = 3600) -> str:
    return CLIENT.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=ttl,
    )

def download_to_temp(key: str) -> str:
    suffix = Path(key).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        CLIENT.download_fileobj(S3_BUCKET, key, tmp)
        return tmp.name

# ────────────────────────────────────────────────────────────────────
# Model + pre-computed embeddings
# ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

@st.cache_resource(show_spinner="⏳ Loading Wav2Vec2 …")
def init_model() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    mdl  = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    return proc, mdl

processor, model = init_model()

def _fname(k: str) -> str:        # strip directories
    return Path(k).name

@st.cache_data(show_spinner="📦 Loading embeddings …")
def load_all_embeddings() -> Dict[str, np.ndarray]:
    obj = CLIENT.get_object(Bucket=S3_BUCKET, Key="all_embeddings.pt")
    buf = io.BytesIO(obj["Body"].read())
    raw = torch.load(buf, map_location="cpu")      # Dict[str, Tensor]
    return {_fname(k): v.numpy() for k, v in raw.items()}

bird_embeddings = load_all_embeddings()

# ────────────────────────────────────────────────────────────────────
# Numeric helpers
# ────────────────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

@torch.inference_mode()
def compute_embedding(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)            # stereo → mono
    if sr != 16_000:
        wav = resample(wav, sr, 16_000)
    wav = wav.to(DEVICE)
    with torch.cuda.amp.autocast(dtype=torch.float16 if torch.cuda.is_available()
                                 else torch.bfloat16):
        out = model(**processor(
            wav.squeeze(),
            sampling_rate=16_000,
            return_tensors="pt").to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# ────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_species_df(species: str) -> pd.DataFrame:
    keys, embs, files = [], [], []
    for k in list_audio_keys(species):
        emb = bird_embeddings.get(_fname(k))
        if emb is None:
            emb = compute_embedding(download_to_temp(k))
        keys.append(k)
        files.append(Path(k).name)
        embs.append(emb)
    cols = [f"dim_{i}" for i in range(len(embs[0]))]
    df   = pd.DataFrame(embs, columns=cols)
    df["file"], df["s3_key"] = files, keys
    return df

@st.cache_resource(show_spinner=False)
def get_reducer(species: str):
    df = get_species_df(species)
    embed_cols = [c for c in df.columns if c.startswith("dim_")]
    reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                   random_state=42).fit(df[embed_cols].values)
    return reducer, df

def run_umap(reducer: UMAP, df_bird: pd.DataFrame,
             user_emb: np.ndarray) -> pd.DataFrame:
    embed_cols  = [c for c in df_bird.columns if c.startswith("dim_")]
    coords_user = reducer.transform(user_emb.reshape(1, -1))
    df_plot     = df_bird.copy()
    df_plot[["umap_x", "umap_y", "umap_z"]] = reducer.embedding_

    df_user = pd.DataFrame(
        {**{c: v for c, v in zip(embed_cols, user_emb)},
         "file": "You",
         "umap_x": coords_user[0, 0],
         "umap_y": coords_user[0, 1],
         "umap_z": coords_user[0, 2]}, index=[0])

    out = pd.concat([df_plot, df_user], ignore_index=True)
    out["type"] = np.where(out["file"] == "You", "User", "Bird")
    return out

# ────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────
ALL_SPECIES = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"]
ss = st.session_state
if "current_species"    not in ss: ss.current_species    = random.choice(ALL_SPECIES)
if "previous_species"   not in ss: ss.previous_species   = []
if "selected_key"       not in ss: ss.selected_key       = None
if "mimic_submitted"    not in ss: ss.mimic_submitted    = False

species = ss.current_species

# ────────────────────────────────────────────────────────────────────
# UI – reference image & audio
# ────────────────────────────────────────────────────────────────────
st.title("Are you good at making bird calls?")

img_key = f"Images/{species}.jpg"
try:
    st.image(CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read())
except Exception:
    st.write("(no image available for this species)")

# choose ≤20-second clip
durations, short_keys = {}, []
for k in list_audio_keys(species):
    try:
        d = get_duration(path=download_to_temp(k))
        durations[k] = d
        if d <= 20:
            short_keys.append(k)
    except Exception:
        continue
if not short_keys:
    short_keys = [min(durations, key=durations.get)] if durations else []
if ss.selected_key not in short_keys:
    ss.selected_key = random.choice(short_keys)
ref_key = ss.selected_key
st.audio(presigned_url(ref_key), format="audio/mpeg")

# ────────────────────────────────────────────────────────────────────
# User recording
# ────────────────────────────────────────────────────────────────────
st.divider(); st.header(f"Try to mimic the {species}!")

recorder_key = f"mimic_{species}_{Path(ref_key).stem}"
try:
    mimic = st.audio_input("Record your attempt here (≤10 s)",
                           key=recorder_key, max_duration=10)
except TypeError:   # older Streamlit
    mimic = st.audio_input("Record your attempt here", key=recorder_key)

if mimic and not ss.mimic_submitted:
    mimic.seek(0)
    raw = mimic.read()
    if not raw:
        st.error("⚠️ Browser captured 0 bytes. Try again or switch browser.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(raw)
        user_path = tmp.name

    ref_emb  = bird_embeddings.get(_fname(ref_key)) \
               or compute_embedding(download_to_temp(ref_key))
    user_emb = compute_embedding(user_path)
    score    = int((cosine_similarity(ref_emb, user_emb) + 1) / 2 * 100)

    ss.mimic_submitted = True
    st.metric("Similarity", f"{score}%")

    with st.spinner("Plotting your call among the bird’s calls …"):
        reducer, df_bird = get_reducer(species)
        df_plot = run_umap(reducer, df_bird, user_emb)
        fig = px.scatter_3d(
            df_plot, x="umap_x", y="umap_y", z="umap_z",
            color="type", hover_name="file",
            color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"})
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Your call is orange; bird calls are olive.")

# ────────────────────────────────────────────────────────────────────
# Navigation buttons
# ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if st.button("🦉 Try a new bird"):
        ss.previous_species.append(species)
        ss.previous_species = ss.previous_species[-3:]
        candidates = [s for s in ALL_SPECIES if s not in ss.previous_species]
        ss.current_species = random.choice(candidates or ALL_SPECIES)
        ss.selected_key = None
        ss.mimic_submitted = False
        ss.pop(recorder_key, None)
        st.rerun()

with col2:
    if ss.mimic_submitted:
        if st.button("🎶 Try the call again"):
            ss.selected_key = random.choice(short_keys)
            ss.mimic_submitted = False
            ss.pop(recorder_key, None)
            st.rerun()
    else:
        st.button("🎶 Try the call again", disabled=True)
