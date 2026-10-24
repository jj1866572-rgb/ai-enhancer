import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance

app = FastAPI(title="Real-ESRGAN AI Enhancer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# বিশ্বসেরা Real-ESRGAN Compact 4x3 মডেল (৪.৮ মেগাবাইট)
MODEL_PATH = "/tmp/realesr_x4v3.onnx"
MODEL_URL = "https://huggingface.co/CoderViking/realesr-general-x4v3-onnx/resolve/main/realesr-general-x4v3.onnx"
session = None

def get_session():
    global session
    if session is None:
        if not os.path.exists(MODEL_PATH):
            req = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as resp, open(MODEL_PATH, 'wb') as f:
                f.write(resp.read())
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    return session

@app.get("/")
@app.get("/api")
@app.get("/api/index.py")
def health():
    return {"status": "online", "model": "Real-ESRGAN AI Enhancer Ready"}

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
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        w, h = img.size

        # সার্ভারের গতি ও র্যামের সুরক্ষার জন্য অপটিমাল রিসাইজিং
        max_dim = 360
        if max(w, h) > max_dim:
            ratio = max_dim / float(max(w, h))
            img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
            w, h = img.size

        # রিয়েল-ESRGAN টেনসর তৈরি
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_tensor = np.expand_dims(img_np, axis=0)

        # ডিপ নিউরাল নেটওয়ার্ক এনহ্যান্সমেন্ট রান করা
        ort_sess = get_session()
        input_name = ort_sess.get_inputs()[0].name
        outputs = ort_sess.run(None, {input_name: img_tensor})

        # আউটপুট ইমেজ রিকভারি
        out_tensor = outputs[0][0]
        out_tensor = np.clip(out_tensor, 0.0, 1.0)
        out_tensor = np.transpose(out_tensor, (1, 2, 0))
        enhanced_img = Image.fromarray((out_tensor * 255.0).astype(np.uint8), mode="RGB")

        # ফেস এবং টেক্সচার আল্ট্রা-শার্প ফিল্টার
        enhancer = ImageEnhance.Sharpness(enhanced_img)
        enhanced_img = enhancer.enhance(1.4)

        out_io = io.BytesIO()
        enhanced_img.save(out_io, format="JPEG", quality=95)
        return Response(content=out_io.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
