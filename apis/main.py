from fastapi import FastAPI
from apis.routers import views
from apis.model_ai import model_store
from dotenv import load_dotenv
import os
import time
import asyncio

load_dotenv()

ACCESSTOKEN_HF=os.getenv("ACCESS_TOKEN_HF")
MODEL_REPO=os.getenv("MODEL_REPO")

app=FastAPI()

async def initialize_resources():
    try:
        start=time.time()
        print("Start loading model ...")
        model=await model_store.load_or_download_model()
        model_store.set_model(model)
        print(f"Model loaded in {time.time() - start:.2f} seconds.")
        
        # start = time.time()
        # print("[DEBUG] Starting CSV loading...")
        # hscode_data = utils.load_csv_from_s3(BUCKET_NAME, CSV_FILE_NAME, CSV_CACHE_PATH)
        # app.state.hscode_data = hscode_data
    except Exception as e:
        print("Error while initialize resources. ",e)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(initialize_resources()) # Running but nt block main flow


app.include_router(views.router)