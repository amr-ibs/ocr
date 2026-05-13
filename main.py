#working
%%writefile main.py

import os
os.environ["FLAGS_use_mkldnn"] = "0"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
import cv2
import numpy as np
from paddleocr import PaddleOCR

app = FastAPI()

# FIXED
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en'
)

def get_crop(image):

    image = cv2.resize(
        image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # FIXED
    result = ocr.ocr(image)

    if not result or not result[0]:
        return None

    boxes = []

    for line in result[0]:

        bbox = line[0]

        for p in bbox:
            boxes.append(p)

    if not boxes:
        return None

    x1 = int(min(p[0] for p in boxes))
    y1 = int(min(p[1] for p in boxes))
    x2 = int(max(p[0] for p in boxes))
    y2 = int(max(p[1] for p in boxes))

    pad = 25

    x1 = max(0, x1+150- pad)
    y1 = max(0, y1+50 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)

    # draw green rectangle
    draw = image.copy()

    cv2.rectangle(
        draw,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # crop boxed region
    crop = draw[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    ok, buffer = cv2.imencode(".jpg", crop)

    if not ok:
        return None

    return buffer


@app.post("/extract-id-crop")
async def extract(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        npimg = np.frombuffer(contents, np.uint8)

        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                content={"error": "Invalid image"},
                status_code=400
            )

        buffer = get_crop(image)

        if buffer is None:
            return JSONResponse(
                content={"error": "No text detected"},
                status_code=400
            )

        return Response(
            content=buffer.tobytes(),
            media_type="image/jpeg"
        )

    except Exception as e:

        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )