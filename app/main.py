from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse

from fastai.vision import *
import torch
from pathlib import Path
from io import BytesIO

app = FastAPI()

path = Path("/app/app")
learner = load_learner(path)


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

@app.get("/html/", response_class=HTMLResponse)
async def read_items():
    return """
    <html>
        <head>
            <title>Some HTML in here</title>
        </head>
        <body>
            <h1>Look ma! HTML!</h1>
        </body>
    </html>
    """

@app.get("/classify-url")
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


@app.post("/upload")
async def upload_file(file: bytes = File(...)):
    return predict_image_from_bytes(file)