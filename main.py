import os
import io
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from PIL import Image
from transformers import pipeline

model = pipeline("image-classification",
                 model= "./model",
                 image_processor= "./model"
                )

app = FastAPI()

CONFIG = {"UPLOAD FOLDER": "static/"}
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

folder_paths = [Path("./static")]
for folder_path in folder_paths:
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()

@app.get("/")
async def home():
    return {
        "status_code": 200,
        "message": "Server is up",
        "data" : None
        }, 200

@app.post("/predict")
async def predict(image: UploadFile):
    img_path = None

    # Checking if the file exists
    if image.filename == "":
        return JSONResponse({
            "status_code": 400,
            "message": "No file uploaded",
            "data": None
        },400)
    
    # Checking if the file is matching the extension
    if image and allowed_file(image.filename):
        content = await image.read()
        new_img = Image.open(io.BytesIO(content))
        img_path = os.path.join(CONFIG["UPLOAD FOLDER"], image.filename)
        new_img.save(img_path)
    else:        
        return JSONResponse({
            "status_code": 400,
            "message": "Invalid file type, only png, jpg, jpeg are allowed",
            "data": None
        },400)
    
    # Classifications Process
    try:
        start_time = time.time()
        image = Image.open(img_path)
        y_pred = model(image, function_to_apply="softmax")
        end_time = time.time()

        result = {
            "label" : y_pred[0]["label"],
            "score" : f"{(y_pred[0]['score']) * 100:.2f} %",
            "time" : f"{(end_time - start_time) * 1000:.3f} ms"
        }
        result["label"] = str(result["label"])
        result["score"] = str(result["score"])

        return JSONResponse({
            "status_code": 200,
            "message": "Classification Success",
            "data": result
        },200)
    
    except Exception as e:
        return JSONResponse({
            "status_code": 400,
            "message": f"Error occurred during classification because of {str(e)}",
            "data" : None
        },400)

