import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil

from fastapi.responses import FileResponse
import os
import tensorflow as tf
import pickle

app = FastAPI(title='Vietnamese Image Captioning VinBigdata Project')

dir_path = os.path.dirname(os.path.realpath(__file__))
tmpPath = os.path.join(dir_path, 'tmp')
if os.path.exists(tmpPath):
    shutil.rmtree(tmpPath)
if not os.path.exists(tmpPath):
    os.mkdir(tmpPath)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Author: VinBigData. Now head over to " \
           "/docs. "


# @app.get("/getCard")
# def getCard():
#     file_path = get_path().path_Card
#     if os.path.exists(file_path):
#         return FileResponse(file_path)
#
#
# @app.get("/getAvatar")
# def getAvatar():
#     file_path = get_path().path_avatar
#     if os.path.exists(file_path):
#         return FileResponse(file_path)


@app.post("/upload")
async def uploadImg(fileUpload: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    filename = fileUpload.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    file_location = f"tmp/{fileUpload.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(fileUpload.file.read())
    print(f"info: file {fileUpload.filename} saved at {file_location}")

    res, attention_plot = evaluate_load(file_location)

    return {
        "result": res,
    }


if __name__ == '__main__':
    def standardize(inputs):
        inputs = tf.strings.lower(inputs)
        return tf.strings.regex_replace(inputs,
                                        r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


    from utilsHandle import *

    host = "0.0.0.0" if os.getenv("DOCKER-SETUP") else "127.0.0.1"

    # Spin up the server!
    uvicorn.run(app, host=host, port=8000)
