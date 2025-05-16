import os
import torch
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from config import species_to_scrape, data_dir


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("🚀 CUDA available:", torch.cuda.is_available())
print("📦 Using device:", device)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
print("📍 Model is on:", next(model.parameters()).device)

def get_embedding(filepath):
    waveform, sr = librosa.load(filepath, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

def embed_and_save_all():
    embeddings = {}
    all_files = []

    for species in species_to_scrape:
        species_dir = os.path.join(data_dir, species)
        for filename in os.listdir(species_dir):
            if filename.endswith(".mp3") or filename.endswith(".wav"):
                all_files.append((species, filename))

    for species, filename in tqdm(all_files, desc="🔊 Embedding bird calls"):
        path = os.path.join(data_dir, species, filename)
        try:
            emb = get_embedding(path)
            key = f"{species}/{filename}"
            embeddings[key] = emb
        except Exception as e:
            print(f"❌ failed on {filename}: {e}")

    save_path = os.path.join(data_dir, "all_embeddings.pt")
    torch.save(embeddings, save_path)
    print(f"\n✅ Saved all embeddings to: {save_path}")

def load_embeddings():
    path = "all_embeddings.pt"
    return torch.load(path)


