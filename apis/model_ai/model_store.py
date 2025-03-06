from bm25s.hf import BM25HF
import os
from huggingface_hub import login, HfFolder
from dotenv import load_dotenv

load_dotenv()

ACCESSTOKEN_HF=os.getenv("ACCESS_TOKEN_HF")
MODEL_REPO=os.getenv("MODEL_REPO")

print("log achf,mr",ACCESSTOKEN_HF,MODEL_REPO)

HfFolder.save_token(ACCESSTOKEN_HF)
login(token=ACCESSTOKEN_HF)  

model = None  

async def load_or_download_model():
    global model
    if model is None:
        try:
            print("Loading model from the hub...")
            login(token=ACCESSTOKEN_HF)  
            model = BM25HF.load_from_hub(MODEL_REPO, load_corpus=True, mmap=True)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            raise
    return model


def set_model(new_model):
    global model
    model = new_model


def get_model():
    global model
    if model is None:
        raise ValueError("Model has not been loaded yet!")
    return model


