import io
import os
import urllib.request
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageEnhance

app = FastAPI(title="Pro AI Natural Image Enhancer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"status": "online", "model": "Pro AI Natural Enhancer Ready"}

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
        raw_img = Image.open(io.BytesIO(contents))
        
        has_alpha = raw_img.mode == "RGBA"
        alpha_channel = None
        
        if has_alpha:
            alpha_channel = raw_img.split()[3]
            img = raw_img.convert("RGB")
        else:
            img = raw_img.convert("RGB")

        w, h = img.size
        
        # ১. রেজোলিউশন বাড়ানো হলো (৫১২px) যাতে স্কিনের আসল পোরস/লোমকূপ মুছে না যায়
        max_dim = 512
        if max(w, h) > max_dim:
            ratio = max_dim / float(max(w, h))
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if alpha_channel:
                alpha_channel = alpha_channel.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = img.size

        # আসল ছবির একটি হাই-কোয়ালিটি ব্যাকআপ রাখা
        original_backup = img.copy()

        # ২. Real-ESRGAN এআই মডেল রান
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        img_tensor = np.expand_dims(img_np, axis=0)

        ort_sess = get_session()
        input_name = ort_sess.get_inputs()[0].name
        outputs = ort_sess.run(None, {input_name: img_tensor})

        out_tensor = outputs[0][0]
        out_tensor = np.clip(out_tensor, 0.0, 1.0)
        out_tensor = np.transpose(out_tensor, (1, 2, 0))
        ai_enhanced = Image.fromarray((out_tensor * 255.0).astype(np.uint8), mode="RGB")

        # ৩. ন্যাচারাল টেক্সচার ব্লেন্ডিং (প্লাস্টিক লুক দূর করার আসল চাবিকাঠি)
        # আসল ছবিকে সমান সাইজে স্কেল করে এআই ছবির সাথে ২০% ব্লেন্ড করা যাতে স্কিন টেক্সচার জীবন্ত থাকে
        orig_upscaled = original_backup.resize(ai_enhanced.size, Image.Resampling.LANCZOS)
        natural_img = Image.blend(ai_enhanced, orig_upscaled, alpha=0.22)

        # ৪. স্মার্ট আনশার্প মাস্ক (Smart Unsharp Mask)
        # এটি ত্বককে মসৃণ না করে শুধুমাত্র চোখ, দাঁত, চুল ও ফ্রেমের লাইনগুলোকে ক্রিস্টাল ক্লিয়ার শার্প করবে
        natural_img = natural_img.filter(ImageFilter.UnsharpMask(radius=1.8, percent=135, threshold=3))

        # ৫. হালকা কালার ও ডেপথ বুস্ট (যাতে ছবি ডাল না লাগে)
        natural_img = ImageEnhance.Color(natural_img).enhance(1.06)
        natural_img = ImageEnhance.Contrast(natural_img).enhance(1.04)

        out_io = io.BytesIO()
        if has_alpha and alpha_channel:
            # আলফা চ্যানেল নিখুঁতভাবে বসানো
            alpha_resized = alpha_channel.resize(natural_img.size, Image.Resampling.LANCZOS)
            natural_img.putalpha(alpha_resized)
            natural_img.save(out_io, format="PNG")
            return Response(content=out_io.getvalue(), media_type="image/png")
        else:
            natural_img.save(out_io, format="JPEG", quality=96)
            return Response(content=out_io.getvalue(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
