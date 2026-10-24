import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from cv2 import dnn_superres
from PIL import Image, ImageEnhance

app = FastAPI(title="AI Image Enhancer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/tmp/FSRCNN_x4.pb"
MODEL_URL = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
sr = None

def get_sr_model():
    global sr
    if sr is None:
        if not os.path.exists(MODEL_PATH):
            req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                f.write(resp.read())
        
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(MODEL_PATH)
        sr.setModel("fsrcnn", 4)
    return sr

# সব রকম রুট যেন Not Found না দেখিয়ে সার্ভার অনলাইন দেখায়
@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
def health():
    return {"status": "online", "message": "AI Image Enhancer is Ready!"}

# অ্যাপ থেকে আসা সমস্ত POST রিকোয়েস্ট নিশ্চিতভাবে গ্রহণ করা
@app.post("/")
@app.post("/api")
@app.post("/enhance")
@app.post("/api/enhance")
@app.post("/api/index.py")
@app.post("/api/index.py/enhance")
async def enhance_image(
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    target = image if image is not None else file
    if target is None:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        contents = await target.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = pil_img.size
        
        # ৪ গুণ এনহ্যান্সের জন্য অপটিমাইজেশন (সর্বোচ্চ ৫০০ পিক্সেল)
        max_dim = 500
        if max(w, h) > max_dim:
            ratio = max_dim / float(max(w, h))
            pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            
        # OpenCV তে কনভার্ট ও ৪ গুণ সুপার-রেজোলিউশন রান
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        model = get_sr_model()
        result_cv = model.upsample(cv_img)
        
        # আবার PIL ইমেজে রূপান্তর
        result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        
        # ছবির ডিটেইল ও শার্পনেস বাড়ানো
        enhancer_sharp = ImageEnhance.Sharpness(result_pil)
        result_pil = enhancer_sharp.enhance(1.8) 
        
        enhancer_color = ImageEnhance.Color(result_pil)
        result_pil = enhancer_color.enhance(1.15)
        
        enhancer_contrast = ImageEnhance.Contrast(result_pil)
        result_pil = enhancer_contrast.enhance(1.05)

        # রেজাল্ট রিটার্ন করা
        out_io = io.BytesIO()
        result_pil.save(out_io, format="JPEG", quality=95)
        return Response(content=out_io.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
