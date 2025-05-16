"""Bird Call Mimic Game
------------------------------------------------
This Streamlit app lets users practise bird calls. For each round it:
1. Chooses a random bird species (from `config.species_to_scrape`).
2.I chose the bird species due to their funky calls, knowledge I have from guiding at Glacier National Park
2. Plays a short reference clip
3. Lets the user record their own attempt.
4. Computes Wav2Vec2 embeddings, cosine similarity (for the score) & a 3-D UMAP visualisation.
"""
# -------------------------------------------------------------------------
# AWS / S3 helpers (Corrected for "Data/" prefix)
# -------------------------------------------------------------------------

DEFAULT_BUCKET = "bird-database" # Default if not in secrets

@st.cache_resource(show_spinner="Connecting to S3...")
def get_s3_client():
    """Initialise boto3 client once per session."""
    try:
        client = boto3.client(
            "s3",
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_REGION", "us-east-1"),
        )
        return client
    except Exception as e:
        st.error(f"S3 Connection Error: {e}. Please check AWS credentials in Streamlit Secrets.")
        st.stop()
        return None

S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)
CLIENT = get_s3_client()

if CLIENT is None:
    st.error("Failed to initialize S3 client. App cannot continue.")
    st.stop()

def list_audio_keys(species: str) -> List[str]:
    """Return all object keys for the given species folder (mp3/wav only),
    assuming audio files are under 'Data/{species}/'."""
    prefix = f"Data/{species}/"
    paginator = CLIENT.get_paginator("list_objects_v2")
    keys = []
    try:
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".mp3", ".wav")):
                    keys.append(key)
    except Exception as e:
        st.error(f"Error listing audio keys from S3 for {species} under '{prefix}': {e}")
    return keys

def presigned_url(key: str, expires_sec: int = 3600) -> str:
    """Generate a temporary, signed URL for public playback of an S3 object."""
    try:
        return CLIENT.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires_sec,
        )
    except Exception as e:
        st.error(f"Error generating presigned URL for {key}: {e}")
        return ""

def download_to_temp(key: str) -> str:
    """Download an S3 object to a NamedTemporaryFile and return its path."""
    suffix = Path(key).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            CLIENT.download_fileobj(S3_BUCKET, key, tmp)
            return tmp.name
    except Exception as e:
        st.error(f"Error downloading {key} to temporary file: {e}")
        return ""

# ... (rest of your script remains the same, but ensure the image loading
#       also accounts for the "Images/" prefix)

# ---------------------------------------------------------------------
# UI - reference image and audio (Corrected for "Images/" prefix)
# ---------------------------------------------------------------------
st.title("Are you good at making bird calls?")

img_key = f"Images/{species}.jpg" # Assumes Images/ folder in S3_BUCKET
try:
    img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()
    st.image(img_bytes)
except Exception: # Catches NoSuchKey, etc.
    st.caption(f"(No image available for {species} at {S3_BUCKET}/{img_key})")

# Choose a reference audio clip
s3_keys_for_species = list_audio_keys(species)
if not s3_keys_for_species:
    st.error(f"No audio files found for {species} in S3 under 'Data/{species}/'. Please check bucket: {S3_BUCKET}.")
    st.stop()

# ... (rest of the audio loading and playback logic)
Key Changes:

list_audio_keys Function: The prefix for listing audio files in S3 is now set to f"Data/{species}/".
Image Loading: The img_key assumes your images are in the root of your bucket under an "Images/" folder. If they are elsewhere, adjust the img_key accordingly.
Error Message: The error message in list_audio_keys is also updated to reflect the "Data/" prefix.
Update your streamlit_birdcalls.py file in your GitHub repository with this corrected list_audio_keys function and the image loading part. This should now correctly look for your audio files in the "Data/" subfolder within your S3 bucket for each species.

Let me know if you still encounter issues after making this change!


Sources





this is my whole file, use like 1/4 of the try adn except blocks and get rid of all of your comments:

"""Bird Call Mimic Game

------------------------------------------------

This Streamlit app lets users practise bird calls. For each round it:

1. Chooses a random bird species (from `config.species_to_scrape`).

2.I chose the bird species due to their funky calls, knowledge I have from guiding at Glacier National Park

2. Plays a short reference clip

3. Lets the user record their own attempt.

4. Computes Wav2Vec2 embeddings, cosine similarity (for the score) & a 3-D UMAP visualisation.

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

import random

import tempfile

from pathlib import Path

from typing import Dict, Tuple, List



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



# Ensure config.py is in your GitHub repository with species_to_scrape defined

from config import species_to_scrape



# -------------------------------------------------------------------------

# AWS / S3 helpers

# -------------------------------------------------------------------------



DEFAULT_BUCKET = "bird-database" # Default if not in secrets



@st.cache_resource(show_spinner="Connecting to S3...")

def get_s3_client():

    """Initialise boto3 client once per session."""

    # Ensure your Streamlit secrets are named e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

    # And optionally AWS_REGION (defaults to "us-east-1" if not found)

    # Or st.secrets["aws"]["AWS_ACCESS_KEY_ID"] if nested. Adjust access accordingly.

    try:

        client = boto3.client(

            "s3",

            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],

            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],

            region_name=st.secrets.get("AWS_REGION", "us-east-1"),

        )

        # Test connection by listing buckets (optional, but good for debug)

        # client.list_buckets() 

        return client

    except Exception as e:

        st.error(f"S3 Connection Error: {e}. Please check AWS credentials in Streamlit Secrets.")

        st.stop() # Stop app if S3 connection fails

        return None # Should be unreachable due to st.stop()



S3_BUCKET = st.secrets.get("S3_BUCKET", DEFAULT_BUCKET)

CLIENT = get_s3_client()



if CLIENT is None: # Should not happen if st.stop() is called in get_s3_client on error

    st.error("Failed to initialize S3 client. App cannot continue.")

    st.stop()



def list_audio_keys(species: str) -> List[str]:

    """Return all object keys for the given species folder (mp3/wav only)."""

    paginator = CLIENT.get_paginator("list_objects_v2")

    keys = []

    try:

        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{species}/"):

            for obj in page.get("Contents", []):

                key = obj["Key"]

                if key.lower().endswith((".mp3", ".wav")):

                    keys.append(key)

    except Exception as e:

        st.error(f"Error listing audio keys from S3 for {species}: {e}")

    return keys



def presigned_url(key: str, expires_sec: int = 3600) -> str:

    """Generate a temporary, signed URL for public playback of an S3 object."""

    try:

        return CLIENT.generate_presigned_url(

            "get_object",

            Params={"Bucket": S3_BUCKET, "Key": key},

            ExpiresIn=expires_sec,

        )

    except Exception as e:

        st.error(f"Error generating presigned URL for {key}: {e}")

        return "" # Return empty string or handle error appropriately



def download_to_temp(key: str) -> str:

    """Download an S3 object to a NamedTemporaryFile and return its path."""

    suffix = Path(key).suffix

    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:

            CLIENT.download_fileobj(S3_BUCKET, key, tmp)

            return tmp.name

    except Exception as e:

        st.error(f"Error downloading {key} to temporary file: {e}")

        return "" # Return empty string or handle error



# -------------------------------------------------------------------------

# Wav2Vec2 model & pre-computed embeddings

# -------------------------------------------------------------------------



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):

    torch.set_float32_matmul_precision("high")



@st.cache_resource(show_spinner="Loading Wav2Vec2 model...")

def init_model() -> Tuple[Wav2Vec2Processor, Wav2Vec2Model]:

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

    _ = model(torch.zeros(1, 16000, device=device))  # warm-up

    return processor, model



@st.cache_data(show_spinner="Fetching pre-computed embeddings...")

def load_all_embeddings() -> Dict[str, np.ndarray]:

    embeddings_key = "all_embeddings.pt" # Ensure this file is at the root of your S3_BUCKET

    try:

        obj = CLIENT.get_object(Bucket=S3_BUCKET, Key=embeddings_key)

        buf = io.BytesIO(obj["Body"].read())

        embeddings = torch.load(buf, map_location="cpu")

        return {k: v.cpu().numpy() for k, v in embeddings.items()}

    except Exception as e:

        st.error(f"Error loading embeddings from S3 ({S3_BUCKET}/{embeddings_key}): {e}")

        st.warning("Proceeding without pre-computed embeddings. This may affect performance if embeddings are computed on-the-fly.")

        return {}



processor, model = init_model()

bird_embeddings = load_all_embeddings()



# -------------------------------------------------------------------------

# Utilities

# -------------------------------------------------------------------------



def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:

    dot = float(np.dot(v1, v2))

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)

    return 0.0 if denom == 0 else dot / denom



@torch.inference_mode()

def compute_embedding(audio_path: str) -> np.ndarray:

    """Load audio from a local path -> Wav2Vec2 -> 1x768 embedding."""

    if not audio_path or not Path(audio_path).exists():

        st.error(f"Audio path for embedding is invalid or file does not exist: {audio_path}")

        return np.array([]) # Return empty array or handle error

    

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:

        waveform = resample(waveform, sr, 16000)

    waveform = waveform.to(device)

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

    

    if device.type == 'cuda':

        with torch.cuda.amp.autocast():

            outputs = model(**inputs)

    else:

        outputs = model(**inputs)

        

    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()



# -------------------------------------------------------------------------

# Data wrangling

# -------------------------------------------------------------------------

@st.cache_data(show_spinner="Preparing species data...")

def get_species_df(species: str) -> pd.DataFrame:

    s3_audio_keys = list_audio_keys(species)

    processed_embeddings = []

    processed_files = []

    processed_s3_keys = []

    temp_files_to_clean = []



    for key in s3_audio_keys:

        emb = bird_embeddings.get(key)

        if emb is None:

            st.info(f"Embedding for {key} not pre-computed. Computing on-the-fly...")

            local_path = download_to_temp(key)

            if not local_path: continue # Skip if download failed

            

            temp_files_to_clean.append(local_path)

            try:

                emb = compute_embedding(local_path)

                if emb.size == 0: continue # Skip if embedding failed

            except Exception as e:

                st.warning(f"Failed to compute embedding for {key}: {e}")

                continue

        

        processed_embeddings.append(emb)

        processed_files.append(Path(key).name)

        processed_s3_keys.append(key)



    for p_path in temp_files_to_clean: # Cleanup temp files

        if Path(p_path).exists():

            try: Path(p_path).unlink()

            except OSError: pass



    if not processed_embeddings:

        # Return an empty DataFrame with expected columns if no data

        return pd.DataFrame(columns=[f"dim_{i}" for i in range(768)] + ["file", "s3_key"])

    

    df = pd.DataFrame(processed_embeddings, columns=[f"dim_{i}" for i in range(processed_embeddings[0].shape[0])])

    df["file"] = processed_files

    df["s3_key"] = processed_s3_keys

    return df



# -------------------------------------------------------------------------

# UMAP

# -------------------------------------------------------------------------

@st.cache_resource(show_spinner="Fitting UMAP reducer...") # UMAP reducer fitting can be slow

def get_reducer(species: str, n_neighbors: int = 15, min_dist: float = 0.1):

    species_df = get_species_df(species)

    if species_df.empty or not any(col.startswith("dim_") for col in species_df.columns):

        st.warning(f"No embedding data to fit UMAP for {species}.")

        return None, species_df



    embed_cols = [c for c in species_df.columns if c.startswith("dim_")]

    reducer = UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit(

        species_df[embed_cols].values

    )

    return reducer, species_df



def run_umap(reducer: UMAP, species_df: pd.DataFrame, user_emb: np.ndarray) -> pd.DataFrame:

    if reducer is None or species_df.empty :

        st.warning("UMAP plot cannot be generated due to missing bird data or reducer.")

        # Create a DF with only user data if possible, or empty if user_emb is also missing

        if user_emb is not None and user_emb.size > 0:

             # Cannot transform user_emb if reducer is None

             st.warning("Cannot place user call in UMAP space without a reducer.")

             return pd.DataFrame([{"file": "You", "type": "User", "umap_x": 0, "umap_y": 0, "umap_z": 0}]) # Placeholder

        return pd.DataFrame()





    embed_cols = [c for c in species_df.columns if c.startswith("dim_")]

    if not embed_cols: # Should not happen if species_df is not empty and from get_species_df

        st.error("Cannot run UMAP: No embedding columns in species data.")

        return species_df



    coords_bird = reducer.embedding_

    coords_user = reducer.transform(user_emb.reshape(1, -1))



    df2 = species_df.copy()

    df2[["umap_x", "umap_y", "umap_z"]] = coords_bird



    user_row_data = dict(zip(embed_cols, user_emb))

    user_row_data.update({

        "file": "You", "s3_key": "N/A-User", # Add s3_key for consistency if needed

        "umap_x": float(coords_user[0, 0]),

        "umap_y": float(coords_user[0, 1]),

        "umap_z": float(coords_user[0, 2]),

    })

    user_df_row = pd.DataFrame([user_row_data])

    df2 = pd.concat([df2, user_df_row], ignore_index=True)

    df2["type"] = df2["file"].apply(lambda f: "User" if f == "You" else "Bird")

    return df2



# -------------------------------------------------------------------------

# Session state init

# -------------------------------------------------------------------------

if not species_to_scrape: # Ensure config.py is not empty

    st.error("`species_to_scrape` in config.py is empty. Please define species.")

    st.stop()



all_species = [s for s in species_to_scrape if s != "Eastern Cattle Eagret"] # Example exclusion

if not all_species:

    st.error("No species available after filtering. Check `species_to_scrape` and exclusions.")

    st.stop()



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



img_key = f"Images/{species}.jpg" # Assumes Images/ folder in S3_BUCKET

try:

    img_bytes = CLIENT.get_object(Bucket=S3_BUCKET, Key=img_key)["Body"].read()

    st.image(img_bytes)

except Exception: # Catches NoSuchKey, etc.

    st.caption(f"(No image available for {species} at {S3_BUCKET}/{img_key})")



# Choose a reference audio clip

s3_keys_for_species = list_audio_keys(species)

if not s3_keys_for_species:

    st.error(f"No audio files found for {species} in S3. Please check bucket: {S3_BUCKET}, prefix: {species}/")

    st.stop()



valid_audio_keys: List[str] = []

audio_durations: Dict[str, float] = {}

temp_duration_check_files = []



with st.spinner(f"Selecting a reference call for {species}..."):

    for key in s3_keys_for_species:

        local_audio_p = download_to_temp(key)

        if not local_audio_p: continue

        temp_duration_check_files.append(local_audio_p)

        try:

            d = get_duration(path=local_audio_p)

            audio_durations[key] = d

            if d <= 20:  # Max duration 20s

                valid_audio_keys.append(key)

        except Exception as e:

            st.warning(f"Could not get duration for {key}. Skipping. Error: {e}")



for p_file in temp_duration_check_files: # Cleanup

    if Path(p_file).exists():

        try: Path(p_file).unlink()

        except OSError: pass



if not valid_audio_keys: # Fallback logic

    if audio_durations:

        st.info("No clips <= 20s found. Using the shortest available clip.")

        valid_audio_keys = [min(audio_durations, key=audio_durations.get)]

    else:

        st.warning("Could not determine clip durations. Using any available clip from S3.")

        valid_audio_keys = s3_keys_for_species # Use all if durations failed



if not valid_audio_keys:

    st.error(f"No suitable audio clips could be selected for {species}.")

    st.stop()



if st.session_state.selected_key not in valid_audio_keys or st.session_state.selected_key is None:

    st.session_state.selected_key = random.choice(valid_audio_keys)



ref_key = st.session_state.selected_key

ref_audio_url = presigned_url(ref_key)



if ref_audio_url:

    audio_suffix = Path(ref_key).suffix.lower()

    audio_format_type = "audio/mpeg" # Default for mp3

    if audio_suffix == ".wav":

        audio_format_type = "audio/wav"

    elif audio_suffix == ".mp3":

        audio_format_type = "audio/mpeg"

    # Add other formats if necessary, or let st.audio infer

    st.audio(ref_audio_url, format=audio_format_type)

    # st.caption(f"Playing: {Path(ref_key).name}") # Optional: show filename

else:

    st.error("Could not load reference audio. Presigned URL generation failed.")



# ---------------------------------------------------------------------

# User recording, similarity score and UMAP

# ---------------------------------------------------------------------

st.divider()

st.header(f"Try to mimic the {species}!")



recorder_widget_key = f"mimic_audio_{species}_{Path(ref_key).stem if ref_key else 'no_ref'}"

user_audio_data = st.audio_input("Record your attempt here:", key=recorder_widget_key)



if user_audio_data and not st.session_state.mimic_submitted:

    user_temp_audio_path = ""

    try:

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_user_audio:

            tmp_user_audio.write(user_audio_data.read())

            user_temp_audio_path = tmp_user_audio.name



        if not Path(user_temp_audio_path).exists() or Path(user_temp_audio_path).stat().st_size == 0:

            st.error("User recording is empty or could not be saved.")

        else:

            if ref_key not in bird_embeddings:

                st.error(f"Reference embedding for {ref_key} not found. Cannot compute similarity.")

                # Fallback: try to compute ref_emb on-the-fly

                # This adds complexity and delay, but makes it more robust if all_embeddings.pt is incomplete

                # st.info("Attempting to compute reference embedding on-the-fly...")

                # ref_dl_path = download_to_temp(ref_key)

                # if ref_dl_path:

                #    ref_emb = compute_embedding(ref_dl_path)

                #    try: Path(ref_dl_path).unlink()

                #    except OSError: pass

                # else:

                #    st.error("Failed to download reference for on-the-fly embedding.")

                #    ref_emb = None # Ensure ref_emb is defined

            else:

                ref_emb = bird_embeddings[ref_key]



            if ref_emb is not None and ref_emb.size > 0:

                user_emb = compute_embedding(user_temp_audio_path)

                if user_emb.size > 0:

                    sim = cosine_similarity(ref_emb, user_emb)

                    score = int((sim + 1) / 2 * 100)

                    st.session_state.mimic_submitted = True

                    st.metric("Similarity Score:", f"{score}%")



                    with st.spinner("Visualizing your call..."):

                        reducer, species_df_for_umap = get_reducer(species)

                        if reducer and not species_df_for_umap.empty:

                            df_plot = run_umap(reducer, species_df_for_umap, user_emb=user_emb)

                            if not df_plot.empty:

                                fig = px.scatter_3d(

                                    df_plot, x="umap_x", y="umap_y", z="umap_z",

                                    color="type", hover_name="file",

                                    color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"}

                                )

                                st.plotly_chart(fig, use_container_width=True)

                                st.caption("Your call is orange; real bird calls are green.")

                        else:

                             st.warning("Could not generate UMAP plot: No bird data or UMAP reducer for this species.")

                else:

                    st.error("Could not compute embedding for your recording.")

            else:

                st.error("Reference embedding is missing or invalid. Cannot proceed with similarity.")



    except Exception as e:

        st.error(f"An error occurred processing your recording: {e}")

        st.session_state.mimic_submitted = False # Reset on error

    finally:

        if user_temp_audio_path and Path(user_temp_audio_path).exists():

            try: Path(user_temp_audio_path).unlink()

            except OSError: pass



# ---------------------------------------------------------------------

# Navigation buttons

# ---------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    if st.button("🦉 Try a new bird"):

        st.session_state.previous_species.append(species)

        st.session_state.previous_species = st.session_state.previous_species[-3:]

        

        candidates = [s for s in all_species if s not in st.session_state.previous_species]

        if not candidates and all_species: # All recently visited

            st.session_state.current_species = random.choice(all_species)

        elif candidates:

            st.session_state.current_species = random.choice(candidates)

        else: # No species available at all (should be caught earlier)

            st.error("No species available to switch to.")

            st.stop()



        st.session_state.selected_key = None

        st.session_state.mimic_submitted = False

        if recorder_widget_key in st.session_state: # Ensure recorder resets

            st.session_state.pop(recorder_widget_key)

        st.rerun()



with col2:

    if st.session_state.mimic_submitted:

        if st.button("🎶 Try this species again"):

            if valid_audio_keys: # From context above

                 st.session_state.selected_key = random.choice(valid_audio_keys) # Pick a new random ref for same species

            st.session_state.mimic_submitted = False

            if recorder_widget_key in st.session_state: # Ensure recorder resets

                 st.session_state.pop(recorder_widget_key)

            st.rerun()

    else:

        st.button("🎶 Try this species again", disabled=True)



Python

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
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from umap import UMAP

from config import species_to_scrape

st.set_page_config(page_title="Are you good at making bird calls?", page_icon="🪶", layout="wide")

DEFAULT_BUCKET = "bird-database"

@st.cache_resource(show_spinner="Connecting to S3...")
def get_s3_client():
    try:
        return boto3.client(
            "s3",
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets.get("AWS_REGION", "us-east-1"),
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
        return torch.load(buf, map_location="cpu").cpu().numpy()
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
        emb = bird_embeddings.get(key)
        if emb is None:
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

with st.spinner(f"Selecting a call for {species}..."):
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
    audio_format = Path(ref_key).suffix.lower()
    st.audio(ref_audio_url, format=f"audio/{audio_format[1:]}")
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
        if ref_key in bird_embeddings:
            ref_embedding = bird_embeddings[ref_key]
            user_embedding = compute_embedding(user_audio_path)
            if user_embedding.size > 0:
                similarity = cosine_similarity(ref_embedding, user_embedding)
                score = int((similarity + 1) / 2 * 100)
                st.session_state.mimic_submitted = True
                st.metric("Similarity Score:", f"{score}%")
                with st.spinner("Visualizing your call..."):
                    reducer, species_df_umap = get_reducer(species)
                    if reducer and not species_df_umap.empty:
                        umap_df = run_umap(reducer, species_df_umap, user_embedding)
                        if not umap_df.empty:
                            fig = px.scatter_3d(umap_df, x="umap_x", y="umap_y", z="umap_z", color="type", hover_name="file", color_discrete_map={"Bird": "#babd8d", "User": "#fa9500"})
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Your call is orange; real bird calls are green.")
        else:
            st.error(f"Reference embedding for {ref_key} not found.")
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
