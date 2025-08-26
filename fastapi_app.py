# fastapi_app.py
# 원단 이미지 업로드 → 지배색 추출 → 26SS 'h' 글씨색 적용(+옵션: 실 Top4) → 결과 이미지 저장 → URL 반환
import os, io, uuid, csv, math
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

# -------------------- 설정 --------------------
OUTPUT_DIR = "outputs"
LOGIC_CSV = os.getenv("LOGIC_CSV", "26SS_logic_final.csv")            # 26개 코드/색 CSV (없어도 동작)
THREADBOOK_CSV = os.getenv("THREADBOOK_CSV", "imjae_threadbook_full.csv")  # 실북 CSV (없어도 동작)
API_KEY = os.getenv("API_KEY")                                        # 배포 시 보안키(로컬 테스트면 비워도 OK)
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
    if not h.startswith("#"): h = "#"+h
    return (int(h[1:3],16), int(h[3:5],16), int(h[5:7],16))

def extract_dominant_rgb(img_pil: Image.Image, k: int = 6) -> Tuple[int,int,int]:
    """원단 지배색 추출(극백/극흑 제외)"""
    im = img_pil.convert("RGB").resize((120,120))
    q = im.quantize(colors=k)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)  # (count, palette_index)
    for cnt, idx in counts:
        r,g,b = pal[3*idx:3*idx+3]
        if not (r>240 and g>240 and b>240) and not (r<15 and g<15 and b<15):
            return (r,g,b)
    idx = counts[0][1]
    return tuple(pal[3*idx:3*idx+3])

def euclid(a: Tuple[int,int,int], b: Tuple[int,int,int]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

# -------------------- 26SS 로직 (여러 스키마 지원) --------------------
def load_26ss_logic(path: str) -> Dict:
    """
    지원 스키마:
      A) code,h_r,h_g,h_b[,fabric_r,fabric_g,fabric_b]
      B) code,h_hex[,fabric_hex]
      C) Thread_No,Thread_Name,R,G,B   ← 이 형식 지원(Thread_No를 code, R/G/B를 h_rgb로 사용)
    반환: {"codes": [...], "h_map": {code: (r,g,b)}, "swatch_map": {code: (r,g,b)}}
    """
    h_map, swatch_map, codes = {}, {}, []
    if not os.path.exists(path):
        for c in ["26SS_08","26SS_09","26SS_12"]:
            h_map[c] = (33,83,145)
            codes.append(c)
        return {"codes": codes, "h_map": h_map, "swatch_map": swatch_map}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        # 구분자 자동 감지(콤마/세미콜론/탭 모두 허용)
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)

        for row in reader:
            r = { (k or "").strip().lower(): (v.strip() if isinstance(v,str) else v) for k,v in row.items() }

            # 형식 C: Thread_No,Thread_Name,R,G,B
            if "thread_no" in r and all(k in r for k in ("r","g","b")):
                code = r["thread_no"]
                try:
                    h_rgb = (int(float(r["r"])), int(float(r["g"])), int(float(r["b"])))
                except Exception:
                    continue
                h_map[code] = h_rgb
                codes.append(code)
                continue

            # 형식 A/B: code + h_rgb/hex (+옵션 fabric)
            if "code" in r:
                code = r["code"]
                # h
                if r.get("h_hex"):
                    try:
                        h_rgb = hex_rgb(r["h_hex"])
                    except Exception:
                        continue
                elif all(k in r for k in ("h_r","h_g","h_b")):
                    try:
                        h_rgb = (int(float(r["h_r"])), int(float(r["h_g"])), int(float(r["h_b"])))
                    except Exception:
                        continue
                else:
                    continue
                h_map[code] = h_rgb
                codes.append(code)
                # fabric(옵션)
                if all(k in r for k in ("fabric_r","fabric_g","fabric_b")):
                    try:
                        fr = int(float(r["fabric_r"])); fg = int(float(r["fabric_g"])); fb = int(float(r["fabric_b"]))
                        swatch_map[code] = (fr,fg,fb)
                    except Exception:
                        pass
                elif r.get("fabric_hex"):
                    try:
                        swatch_map[code] = hex_rgb(r["fabric_hex"])
                    except Exception:
                        pass

    return {"codes": sorted(set(codes)), "h_map": h_map, "swatch_map": swatch_map}

# -------------------- 실북 Top4 (선택) --------------------
def load_threadbook(path: str) -> Optional[List[Dict]]:
    if not os.path.exists(path): return None
    try:
        rows: List[Dict] = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096); f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                r = { (k or "").strip().lower(): (v.strip() if isinstance(v,str) else v) for k,v in row.items() }
                if not all(k in r for k in ("thread_no","thread_name","r","g","b")):
                    continue
                rows.append({
                    "Thread_No": r["thread_no"],
                    "Thread_Name": r["thread_name"],
                    "R": int(float(r["r"])), "G": int(float(r["g"])), "B": int(float(r["b"]))
                })
        return rows if rows else None
    except Exception:
        return None

THREADBOOK = load_threadbook(THREADBOOK_CSV)

def nearest_threads(target_rgb: Tuple[int,int,int], rows: List[Dict], top_n: int = 4) -> List[Dict]:
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

def looks_suspicious(threads: List[Dict]) -> bool:
    # 채도 낮은 색이 3개 이상이면 의심(단순 가드)
    def sat(rgb):
        r,g,b = [c/255 for c in rgb]
        mx, mn = max(r,g,b), min(r,g,b)
        return 0 if mx==0 else (mx-mn)/mx
    return sum(1 for t in threads if sat(t["RGB"]) < 0.15) >= 3

# -------------------- 렌더러 --------------------
def render_preview(fabric_rgb: Tuple[int,int,int], threads: List[Dict], h_rgb: Tuple[int,int,int],
                   W: int = 1600, H: int = 900, bg=(255,255,255)) -> Image.Image:
    margin = 40; left_w = 800
    img = Image.new("RGB",(W,H),bg)
    d = ImageDraw.Draw(img)
    font_big   = _font(520)
    font_title = _font(36)
    font_small = _font(24)

    # 좌측: 원단 배경 + 'h'
    left = (margin, margin, margin+left_w, H-margin)
    d.rectangle(left, fill=fabric_rgb)
    tx, ty, bx, by = d.textbbox((0,0), "h", font=font_big)
    d.text((margin + left_w//2 - (bx-tx)//2, margin + (H-2*margin)//2 - (by-ty)//2),
           "h", font=font_big, fill=h_rgb)
    d.text((margin+16, H-margin-46),
           f"Fabric RGB {fabric_rgb}  {rgb_hex(fabric_rgb)}",
           font=font_title, fill=(255,255,255))

    # 우측: 2x2 칩
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

# -------------------- 보안키(선택) --------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# -------------------- FastAPI 앱 --------------------
app = FastAPI(title="HAZZYS Embroidery Recommender API", version="1.0.0")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/")
def root():
    return {"message": "HAZZYS Embroidery API. See /docs for usage."}

@app.get("/health")
def health():
    return {"ok": True}

def _pick_code_auto(
    fabric_rgb: Tuple[int,int,int],
    codes: List[str],
    swatch_map: Dict[str,Tuple[int,int,int]],
    h_map: Dict[str,Tuple[int,int,int]]
) -> str:
    # 1순위: 스와치(fabric_*)가 있으면 그 기준으로 최근접
    if swatch_map:
        candidates = []
        for c in codes:
            if c in swatch_map:
                candidates.append((euclid(fabric_rgb, swatch_map[c]), c))
        if candidates:
            return min(candidates, key=lambda x: x[0])[1]
    # 2순위: 스와치가 없으면 각 코드의 h 색과 최근접
    candidates = []
    for c in codes:
        if c in h_map:
            candidates.append((euclid(fabric_rgb, h_map[c]), c))
    if candidates:
        return min(candidates, key=lambda x: x[0])[1]
    # 최후의 보루
    return codes[0] if codes else "26SS_09"

@app.post("/render")
async def render_endpoint(
    request: Request,
    fabric_image: UploadFile = File(..., description="Fabric image (png/jpg)"),
    ss_code: str = Form("auto", description="26SS code or 'auto'"),
    out_format: str = Form("png", description="'png' or 'jpg'"),
    width: int = Form(1600), height: int = Form(900),
    include_top4: bool = Form(True),
    _ok: bool = Depends(require_api_key)
):
    # 26SS 로직 로드
    logic = load_26ss_logic(LOGIC_CSV)
    codes, h_map, swatch_map = logic["codes"], logic["h_map"], logic["swatch_map"]

    # CSV가 없거나 형식이 달라도 최소 동작(안전 폴백)
    if not codes:
        codes = ["26SS_08","26SS_09","26SS_12"]
        h_map = {c: (33,83,145) for c in codes}
        swatch_map = {}

    # 이미지 읽기
    raw = await fabric_image.read()
    try:
        pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Invalid image file")

    fabric_rgb = extract_dominant_rgb(pil)

    # 코드 선택
    pick = (ss_code or "auto").strip()
    if pick == "auto" or pick not in h_map:
        pick = _pick_code_auto(fabric_rgb, codes, swatch_map, h_map)
    h_rgb = h_map[pick]

    # 실 Top4 (CSV 없으면 h_rgb 파생 4종으로 대체)
    threads: List[Dict] = []
    tb = THREADBOOK
    if include_top4 and tb:
        try:
            top4 = nearest_threads(h_rgb, tb, top_n=4)
            if not looks_suspicious(top4):
                threads = top4
        except Exception:
            pass
    if include_top4 and not threads:
        threads = [
            {"Thread_No":"—","Thread_Name":"Nearest-1","RGB":h_rgb},
            {"Thread_No":"—","Thread_Name":"Nearest-2","RGB":tuple(max(min(int(c*0.90),255),0) for c in h_rgb)},
            {"Thread_No":"—","Thread_Name":"Nearest-3","RGB":tuple(max(min(int(c*0.80),255),0) for c in h_rgb)},
            {"Thread_No":"—","Thread_Name":"Nearest-4","RGB":tuple(max(min(int(c*0.70),255),0) for c in h_rgb)},
        ]

    # 렌더 & 저장
    img = render_preview(fabric_rgb, threads if include_top4 else [], h_rgb, W=width, H=height)
    ext = "png" if out_format.lower() not in ["jpg","jpeg"] else "jpg"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{ext}"
    save_path = os.path.join(OUTPUT_DIR, filename)
    img.convert("RGB").save(save_path, ext.upper())

    # URL
    base = str(request.base_url).rstrip("/")
    url = f"{base}/outputs/{filename}"

    meta = {
        "fabric_rgb": {"rgb": fabric_rgb, "hex": rgb_hex(fabric_rgb)},
        "26ss_code": pick,
        "h_rgb": {"rgb": h_rgb, "hex": rgb_hex(h_rgb)},
        "include_top4": include_top4,
        "threads": [{"no":t["Thread_No"], "name":t["Thread_Name"],
                     "rgb":t["RGB"], "hex": rgb_hex(t["RGB"])} for t in threads]
    }
    return JSONResponse({"image_url": url, "meta": meta})
