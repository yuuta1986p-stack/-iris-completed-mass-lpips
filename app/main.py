from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch, lpips
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Iris Completed Mass LPIPS v2")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = lpips.LPIPS(net="alex").to(device).eval()

def crop_iris(data: bytes, cx: float, cy: float, radius: float, size=256):
    im = Image.open(BytesIO(data)).convert("L")
    w,h = im.size
    x,y,r = cx*w, cy*h, radius*min(w,h)
    if r <= 2: raise ValueError("虹彩半径が小さすぎます")
    box=(x-r,y-r,x+r,y+r)
    return im.crop(box).resize((size,size), Image.Resampling.LANCZOS)

def completed_mass(im, sigma):
    arr=np.asarray(im,dtype=np.float32); n=arr.shape[0]
    yy,xx=np.ogrid[:n,:n]; c=(n-1)/2; r=n*.49
    mask=(xx-c)**2+(yy-c)**2<=r*r
    mean=float(arr[mask].mean()); arr[~mask]=mean
    return Image.fromarray(np.uint8(np.clip(arr,0,255))).filter(ImageFilter.GaussianBlur(radius=sigma))

def tensorize(im):
    a=np.asarray(im,dtype=np.float32)/127.5-1
    a=np.stack([a,a,a],0)
    return torch.from_numpy(a).unsqueeze(0).to(device)

@app.get("/")
async def root(): return FileResponse(static_dir/"index.html")
@app.get("/manifest.webmanifest")
async def manifest(): return FileResponse(static_dir/"manifest.webmanifest",media_type="application/manifest+json")
@app.get("/api/health")
async def health(): return {"ok":True,"engine":"LPIPS","net":"alex","device":str(device)}

@app.post("/api/compare")
async def compare(
    a: UploadFile=File(...), b: UploadFile=File(...),
    acx: float=Form(...), acy: float=Form(...), ar: float=Form(...),
    bcx: float=Form(...), bcy: float=Form(...), br: float=Form(...)
):
    try:
        ia=crop_iris(await a.read(),acx,acy,ar); ib=crop_iris(await b.read(),bcx,bcy,br)
    except Exception as e: raise HTTPException(400,str(e))
    vals=[]
    with torch.inference_mode():
        for s in (16.,32.,48.):
            d=float(model(tensorize(completed_mass(ia,s)),tensorize(completed_mass(ib,s))).item())
            vals.append({"sigma":int(s),"lpips_distance":d})
    return {"engine":"LPIPS","net":"alex","scales":vals,
            "mean_lpips_distance":float(np.mean([v["lpips_distance"] for v in vals])),
            "official_similarity_score":None,"language_band":None,
            "note":"LPIPS公式には100点換算・Completed Mass言語帯境界はありません。"}
