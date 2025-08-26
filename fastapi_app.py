# fastapi_app.py
import os, io, uuid, csv, math, re, base64
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

# -------------------- 설정 --------------------
OUTPUT_DIR = "outputs"
LOGIC_CSV = os.getenv("LOGIC_CSV", "26SS_logic_final.csv")
THREADBOOK_CSV = os.getenv("THREADBOOK_CSV", "imjae_threadbook_full.csv")
API_KEY = os.getenv("API_KEY")

USE_CURATED_H = os.getenv("USE_CURATED_H", "1").lower() in ("1","true","yes")
CURATED_H: Dict[str, Tuple[int,int,int]] = {
    "26SS_08": (33,83,145),
    "26SS_09": (33,83,145),
    "26SS_12": (33,83,145),
}

FIXED_RECS: Dict[str, List[Dict]] = {
    c: [
        {"Thread_No":"14846","Thread_Name":"ELECTRIC GREY","RGB":(140,140,140)},
        {"Thread_No":"16292","Thread_Name":"SECRET BLUE","RGB":(18,69,139)},
        {"Thread_No":"13643","Thread_Name":"MAGIC BLUE","RGB":(25,77,142)},
        {"Thread_No":"16296","Thread_Name":"LOYAL BLUE","RGB":(33,83,145)},
    ] for c in ("26SS_08","26SS_09","26SS_12")
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 글꼴 --------------------
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
def _font(sz: int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, sz)
            except:
                pass
    return ImageFont.load_default()

# -------------------- 유틸 --------------------
def rgb_hex(c: Tuple[int,int,int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*c)

def hex_rgb(h: str) -> Tuple[int,int,int]:
    h = h.strip()
    if not h.startswith("#"): h = "#" + h
    return (int(h[1:3],16), int(h[3:5],16), int(h[5:7],16))

def extract_dominant_rgb(img_pil: Image.Image, k: int = 6) -> Tuple[int,int,int]:
    im = img_pil.convert("RGB").resize((120,120))
    q = im.quantize(colors=k)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)
    for cnt, idx in counts:
        r,g,b = pal[3*idx:3*idx+3]
        if not (r>240 and g>240 and b>240) and not (r<15 and g<15 and b<15):
            return (r,g,b)
    idx = counts[0][1]
    return tuple(pal[3*idx:3*idx+3])

def euclid(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def normalize_ss_code(s: str) -> str:
    s = (s or "").strip().upper()
    if s == "AUTO":
        return "auto"
    nums = re.findall(r"\d+", s)
    if nums:
        n = int(nums[-1])
        return f"26SS_{n:02d}"
    return s

# -------------------- CSV 로드 --------------------
def load_26ss_logic(path: str):
    h_map, swatch_map, codes = {}, {}, []
    if not os.path.exists(path):
        for c in ("26SS_08","26SS_09","26SS_12"):
            h_map[c] = (33,83,145)
            codes.append(c)
        return {"codes":codes, "h_map":h_map, "swatch_map":swatch_map}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            r = { (k or "").strip().lower(): (v.strip() if isinstance(v,str) else v) for k,v in row.items() }
            if "thread_no" in r and all(k in r for k in ("r","g","b")):
                code = r["thread_no"]
                try:
                    h_rgb = (int(float(r["r"])), int(float(r["g"])), int(float(r["b"])))
                except:
                    continue
                h_map[code] = h_rgb
                codes.append(code)
                continue
            if "code" in r:
                code = r["code"]
                if r.get("h_hex"):
                    try:
                        h_rgb = hex_rgb(r["h_hex"])
                    except:
                        continue
                elif all(k in r for k in ("h_r","h_g","h_b")):
                    try:
                        h_rgb = (int(float(r["h_r"])), int(float(r["h_g"])), int(float(r["h_b"])))
                    except:
                        continue
                else:
                    continue
                h_map[code] = h_rgb
                codes.append(code)
                if all(k in r for k in ("fabric_r","fabric_g","fabric_b")):
                    try:
                        swatch_map[code] = (
                            int(float(r["fabric_r"])),
                            int(float(r["fabric_g"])),
                            int(float(r["fabric_b"]))
                        )
                    except:
                        pass
                elif r.get("fabric_hex"):
                    try:
                        swatch_map[code] = hex_rgb(r["fabric_hex"])
                    except:
                        pass
    return {"codes":sorted(set(codes)), "h_map":h_map, "swatch_map":swatch_map}

def load_threadbook(path: str):
    if not os.path.exists(path):
        return None
    try:
        rows = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096); f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                r = { (k or "").strip().lower(): (v.strip() if isinstance(v,str) else v) for k,v in row.items() }
                if not all(k in r for k in ("thread_no","thread_name","r","g","b")):
                    continue
                rows.append({
                    "Thread_No": r["thread_no"],
                    "Thread_Name": r["thread_name"],
                    "R": int(float(r["r"])),
                    "G": int(float(r["g"])),
                    "B": int(float(r["b"])),
                })
        return rows if rows else None
    except:
        return None

THREADBOOK = load_threadbook(THREADBOOK_CSV)

def nearest_threads(target_rgb, rows, top_n=4):
    scored = []
    for r in rows:
        dist = euclid(target_rgb, (r["R"], r["G"], r["B"]))
        scored.append((dist, r))
    scored.sort(key=lambda x: x[0])
    out = []
    for _, row in scored[:top_n]:
        out.append({
            "Thread_No": row["Thread_No"],
            "Thread_Name": row["Thread_Name"],
            "RGB": (row["R"], row["G"], row["B"])
        })
    return out

def looks_suspicious(threads):
    def sat(rgb):
        r,g,b = [c/255 for c in rgb]
        mx, mn = max(r,g,b), min(r,g,b)
        return 0 if mx==0 else (mx-mn)/mx
    return sum(1 for t in threads if sat(t["RGB"]) < 0.15) >= 3

# -------------------- 렌더링 --------------------
def render_preview(fabric_rgb, threads, h_rgb, W=1600, H=900, bg=(255,255,255)):
    margin, left_w = 40, 800
    img = Image.new("RGB", (W,H), bg)
    d = ImageDraw.Draw(img)
    font_big = _font(520)
    font_title = _font(36)
    font_small = _font(24)
    left = (margin, margin, margin+left_w, H-margin)
    d.rectangle(left, fill=fabric_rgb)
    tx, ty, bx, by = d.textbbox((0,0), "h", font=font_big)
    d.text(
        (margin + left_w//2 - (bx-tx)//2, margin + (H-2*margin)//2 - (by-ty)//2),
        "h", font=font_big, fill=h_rgb)
    d.text((margin+16, H-margin-46),
           f"Fabric RGB {fabric_rgb}  {rgb_hex(fabric_rgb)}",
           font=font_title, fill=(255,255,255))
    right_w = W - left_w - 3*margin
    grid_gap = margin//2
    cell_w = (right_w - grid_gap)//2
    cell_h = (H - 3*margin)//2
    start_x = margin*2 + left_w
    start_y = margin
    for i, t in enumerate(threads[:4]):
        r = i//2; c = i%2
        x0 = start_x + c*(cell_w + grid_gap)
        y0 = start_y + r*(cell_h + margin)
        x1 = x0 + cell_w; y1 = y0 + cell_h
        d.rectangle((x0, y0, x1, y1), fill=t["RGB"])
        strip_h = 120
        d.rectangle((x0, y1-strip_h, x1, y1), fill=(255,255,255))
        d.text((x0+14, y1-strip_h+12),
               f"#{t['Thread_No']}  {t['Thread_Name']}",
               font=font_title, fill=(0,0,0))
        d.text((x0+14, y1-strip_h+58),
               f"RGB {t['RGB']}  {rgb_hex(t['RGB'])}",
               font=font_small, fill=(0,0,0))
    return img

# -------------------- 보안키 --------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# -------------------- FastAPI 앱 --------------------
app = FastAPI(title="HAZZYS Embroidery Recommender API", version="1.0.0")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
def root():
    return {"message": "HAZZYS Embroidery API. Use /docs or POST /render."}

@app.get("/health")
def health():
    return {"ok": True}

def _pick_code_auto(fabric_rgb, codes, swatch_map, h_map):
    if swatch_map:
        candidates = [(euclid(fabric_rgb, swatch_map[c]), c) for c in codes if c in swatch_map]
        if candidates:
            return min(candidates, key=lambda x: x[0])[1]
    candidates = [(euclid(fabric_rgb, h_map[c]), c) for c in codes if c in h_map]
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]
    return codes[0] if codes else "26SS_09"

@app.post("/render")
async def render_endpoint(
    request: Request,
    fabric_image: UploadFile = File(..., description="Fabric image"),
    ss_code: str = Form("auto", description="26SS code"),
    out_format: str = Form("png"),
    width: int = Form(1600),
    height: int = Form(900),
    include_top4: bool = Form(True),
    use_fixed: bool = Form(True),
    _ok: bool = Depends(require_api_key)
):
    logic = load_26ss_logic(LOGIC_CSV)
    codes, h_map, swatch_map = logic["codes"], logic["h_map"], logic["swatch_map"]
    if not codes:
        codes, h_map, swatch_map = ["26SS_08","26SS_09","26SS_12"], {c:(33,83,145) for c in ("26SS_08","26SS_09","26SS_12")}, {}
    if USE_CURATED_H:
        for c, rgb in CURATED_H.items():
            h_map[c] = rgb
            if c not in codes:
                codes.append(c)
    raw = await fabric_image.read()
    try:
        pil = Image.open(io.BytesIO(raw))
    except:
        raise HTTPException(400, "Invalid image file")
    fabric_rgb = extract_dominant_rgb(pil)
    raw_code = normalize_ss_code(ss_code)
    if raw_code == "auto" or raw_code not in h_map:
        pick = _pick_code_auto(fabric_rgb, codes, swatch_map, h_map)
    else:
        pick = raw_code
    h_rgb = CURATED_H.get(pick, h_map[pick])
    threads = []
    if use_fixed and pick in FIXED_RECS:
        threads = FIXED_RECS[pick]
    if not threads and include_top4 and THREADBOOK:
        try:
            top4 = nearest_threads(h_rgb, THREADBOOK, top_n=4)
            if not looks_suspicious(top4):
                threads = top4
        except:
            threads = []
    if include_top4 and not threads:
        threads = [
            {"Thread_No":"—","Thread_Name":"Nearest-1","RGB":h_rgb},
            {"Thread_No":"—","Thread_Name":"Nearest-2","RGB":tuple(max(min(int(c*0.90),255),0) for c in h_rgb)},
            {"Thread_No":"—","Thread_Name":"Nearest-3","RGB":tuple(max(min(int(c*0.80),255),0) for c in h_rgb)},
            {"Thread_No":"—","Thread_Name":"Nearest-4","RGB":tuple(max(min(int(c*0.70),255),0) for c in h_rgb)},
        ]

    img = render_preview(fabric_rgb, threads if include_top4 else [], h_rgb, W=width, H=height)
    buf = io.BytesIO()
    img = img.resize((600, 338))  # → 축소하여 Base64 사이즈 줄임
    img.convert("RGB").save(buf, "JPEG", quality=75)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    summary_table = (
        "| 항목 | 값 |\n"
        "|------|----|\n"
        f"| fabric_rgb | {fabric_rgb}, {rgb_hex(fabric_rgb)} |\n"
        f"| 26ss_code | {pick} |\n"
        f"| h_rgb | {h_rgb}, {rgb_hex(h_rgb)} |\n"
        f"| include_top4 | {include_top4} |\n"
        f"| use_fixed | {use_fixed} |\n"
        f"| threads | " + ", ".join(
            f"{t['Thread_No']} {t['Thread_Name']} ({rgb_hex(t['RGB'])})"
            for t in threads
        ) + " |"
    )
    meta = {
        "fabric_rgb": {"rgb": fabric_rgb, "hex": rgb_hex(fabric_rgb)},
        "26ss_code": pick,
        "h_rgb": {"rgb": h_rgb, "hex": rgb_hex(h_rgb)},
        "include_top4": include_top4,
        "use_fixed": use_fixed,
        "threads": [{"no":t["Thread_No"], "name":t["Thread_Name"], "rgb":t["RGB"], "hex": rgb_hex(t["RGB"])} for t in threads]
    }
    return JSONResponse({
        "image_data": data_uri,
        "summary_table": summary_table,
        "meta": meta
    })


