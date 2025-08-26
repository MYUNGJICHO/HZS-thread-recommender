# fastapi_app.py
import os, io, uuid, csv, math, re, base64
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

# --- 설정 ---
OUTPUT_DIR = "outputs"
LOGIC_CSV = os.getenv("LOGIC_CSV", "26SS_logic_final.csv")
THREADBOOK_CSV = os.getenv("THREADBOOK_CSV", "imjae_threadbook_full.csv")
API_KEY = os.getenv("API_KEY")
USE_CURATED_H = os.getenv("USE_CURATED_H", "1").lower() in ("1","true","yes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CURATED_H = {
    "26SS_08": (33,83,145),
    "26SS_09": (33,83,145),
    "26SS_12": (33,83,145),
}

FIXED_RECS = {
    c: [
        {"Thread_No":"14846","Thread_Name":"ELECTRIC GREY","RGB":(140,140,140)},
        {"Thread_No":"16292","Thread_Name":"SECRET BLUE","RGB":(18,69,139)},
        {"Thread_No":"13643","Thread_Name":"MAGIC BLUE","RGB":(25,77,142)},
        {"Thread_No":"16296","Thread_Name":"LOYAL BLUE","RGB":(33,83,145)},
    ] for c in ("26SS_08","26SS_09","26SS_12")
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
def _font(sz:int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except:
                pass
    return ImageFont.load_default()

# --- 유틸 ---
def rgb_hex(c): return "#{:02X}{:02X}{:02X}".format(*c)
def hex_rgb(h):
    if not h.strip().startswith("#"): h = "#" + h
    return (int(h[1:3],16), int(h[3:5],16), int(h[5:7],16))
def euclid(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def normalize_ss_code(s: str) -> str:
    s = (s or "").strip().upper()
    if s == "AUTO": return "auto"
    nums = re.findall(r"\d+", s)
    if nums:
        return f"26SS_{int(nums[-1]):02d}"
    return s

# CSV 로딩 (생략 가능 코드는 이전 내용 참고)
def load_26ss_logic(path): ...
def load_threadbook(path): ...
THREADBOOK = load_threadbook(THREADBOOK_CSV)
def nearest_threads(target, rows, top_n=4): ...
def looks_suspicious(threads): ...

def render_preview(fabric_rgb, threads, h_rgb, W=1600, H=900, bg=(255,255,255)):
    margin = 40; left_w = 800
    img = Image.new("RGB",(W,H),bg)
    d = ImageDraw.Draw(img)
    font_big = _font(520); font_title = _font(36); font_small = _font(24)
    # Left: fabric + 'h'
    d.rectangle((margin, margin, margin+left_w, H-margin), fill=fabric_rgb)
    tx, ty, bx, by = d.textbbox((0,0), "h", font=font_big)
    d.text((margin + left_w//2 - (bx-tx)//2,
            margin + (H-2*margin)//2 - (by-ty)//2),
            "h", font=font_big, fill=h_rgb)
    d.text((margin+16, H-margin-46),
           f"Fabric RGB {fabric_rgb}  {rgb_hex(fabric_rgb)}",
           font=font_title, fill=(255,255,255))
    # Right: Top4 chips
    right_w = W - left_w - 3*margin
    grid_gap = margin//2
    cell_w = (right_w - grid_gap)//2
    cell_h = (H - 3*margin)//2
    start_x = 2*margin + left_w
    start_y = margin
    for i, t in enumerate(threads[:4]):
        r = i//2; c = i%2
        x0 = start_x + c*(cell_w + grid_gap)
        y0 = start_y + r*(cell_h + margin)
        x1 = x0 + cell_w; y1 = y0 + cell_h
        d.rectangle((x0,y0,x1,y1), fill=t["RGB"])
        strip_h = 120
        d.rectangle((x0, y1-strip_h, x1, y1), fill=(255,255,255))
        d.text((x0+14, y1-strip_h+12),
               f"#{t['Thread_No']}  {t['Thread_Name']}",
               font=font_title, fill=(0,0,0))
        d.text((x0+14, y1-strip_h+58),
               f"RGB {t['RGB']}  {rgb_hex(t['RGB'])}",
               font=font_small, fill=(0,0,0))
    return img

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return True

# --- FastAPI 앱 ---
app = FastAPI(title="HAZZYS Embroidery Recommender", version="1.0.0")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
def root(): return {"message": "Use /docs or /render"}

@app.get("/health")
def health(): return {"ok": True}

@app.post("/render")
async def render_endpoint(request: Request,
    fabric_image: UploadFile = File(...), ss_code: str = Form("auto"),
    out_format: str = Form("png"), width: int = Form(1600),
    height: int = Form(900), include_top4: bool = Form(True),
    use_fixed: bool = Form(True), _ok: bool = Depends(require_api_key)
):
    logic = load_26ss_logic(LOGIC_CSV)
    codes, h_map, swatch_map = logic["codes"], logic["h_map"], logic["swatch_map"]
    if USE_CURATED_H:
        for c, rgb in CURATED_H.items():
            h_map[c] = rgb
            if c not in codes: codes.append(c)
    raw = await fabric_image.read()
    try:
        pil = Image.open(io.BytesIO(raw))
    except:
        raise HTTPException(400, "Invalid image file")
    fabric_rgb = render_preview(extract_dominant_rgb(pil), [], (0,0,0)).getpixel((10,10))  # placeholder
    raw_code = normalize_ss_code(ss_code)
    pick = raw_code if raw_code != "auto" and raw_code in h_map else _pick_code_auto(fabric_rgb, codes, swatch_map, h_map)
    h_rgb = h_map.get(pick, (0,0,0))
    threads = FIXED_RECS[pick] if use_fixed and pick in FIXED_RECS else nearest_threads(h_rgb, THREADBOOK, 4)
    img = render_preview(fabric_rgb, threads, h_rgb, W=width, H=height)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, out_format.upper())
    data_uri = f"data:image/{out_format};base64," + base64.b64encode(buf.getvalue()).decode()
    summary = (
        "| 항목 | 값 |\n"
        "|------|----|\n"
        f"| fabric_rgb | {fabric_rgb} / {rgb_hex(fabric_rgb)} |\n"
        f"| 26ss_code | {pick} |\n"
        f"| h_rgb | {h_rgb} / {rgb_hex(h_rgb)} |\n"
        f"| include_top4 | {include_top4} |\n"
        f"| use_fixed | {use_fixed} |\n"
        f"| threads | " + ", ".join(f"{t['Thread_No']} {t['Thread_Name']} ({rgb_hex(t['RGB'])})" for t in threads) + " |"
    )
    return JSONResponse({
        "image_data": data_uri,
        "summary_table": summary,
        "meta": {
            "fabric_rgb": {"rgb": list(fabric_rgb), "hex": rgb_hex(fabric_rgb)},
            "26ss_code": pick,
            "h_rgb": {"rgb": list(h_rgb), "hex": rgb_hex(h_rgb)},
            "include_top4": include_top4,
            "use_fixed": use_fixed,
            "threads": threads
        }
    })
