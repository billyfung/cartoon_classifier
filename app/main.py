from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse
from fastai.vision import *
import torch

from pathlib import Path
from io import BytesIO
import aiohttp, asyncio
import logging

MODEL_URL = 'https://drive.google.com/u/0/uc?id=1-Cw70xsI7RBKIFWYvGkElUbW3RZUtacw&export=download'
MODEL_FILENAME = 'export'

app = FastAPI()

path = Path(__file__).parent

@app.on_event("startup")
async def setup_learner():
    await download_file(MODEL_URL, path/f'{MODEL_FILENAME}.pkl')
    global learner
    learner = load_learner(path)

async def download_file(url, dest):
    if dest.exists(): 
        global learner
        learner = load_learner(path)
        return
    async with aiohttp.ClientSession() as session:
        logging.info('getting model file')
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return {
        "predictions": sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    }

@app.get("/", response_class=HTMLResponse)
async def show_form():
    return """
    <html>
        <head>
            <title>Upload a cartoon</title>
        </head>
        <body>
            <form action="/upload" method="post" enctype="multipart/form-data">
                Select image to upload:
                <input type="file" name="file">
                <input type="submit" value="Upload Image">
            </form>
            Or submit a URL:
            <form action="/classify-url" method="get">
                <input type="url" name="url">
                <input type="submit" value="Fetch and analyze image">
            </form>
        </body>
    </html>
    """

@app.get("/classify-url")
async def classify_url(url: str):
    logging.info(f'getting url')
    bytes = await get_bytes(url)
    return predict_image_from_bytes(bytes)

@app.post("/upload")
async def upload_file(file: bytes = File(...)):
    return predict_image_from_bytes(file)