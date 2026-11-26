"""
IV monitoring demo: nurse dashboard (Arabic, RTL).

    streamlit run app/app.py

Board of 6 beds -> open a bed -> upload drip video + bag photo + label photo + prescription
-> analyse with Jana's measure_rate and Hasan's bag_level / check_label -> result on the board.
The nurse sees only what she acts on. Analysis errors are printed in the terminal for the team.
"""
import copy
import io
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import backends
from iv_logic import APPROX, EXAMPLE_BEDS, TONES, evaluate_saved

st.set_page_config(page_title="مراقبة المحلول الوريدي", page_icon="💧", layout="wide",
                   initial_sidebar_state="expanded")

BED_IDS = [1, 2, 3, 4, 5, 6]
FACTORS = [10, 15, 20, 60]
KNOWN_DRUGS = ["Normal Saline 0.9%", "Dextrose 5%", "Ringer Lactate", "محلول ملحي 0.9%", "جلوكوز 5%", "رينجر لاكتات"]
DEFAULT_RX = {"drug": KNOWN_DRUGS[0], "rx_mlh": 100, "drop_factor": 20}

# --------------------------------------------------------------- style
CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root { --ink:#0F1B2D; --ink-2:#4A5868; --line:#E3E8EE; --page:#F2F4F7; --navy:#0E2A3F; --blue:#0A5C8A; }
html, body, .stApp, button, input, textarea, select, p, span, div, label, h1, h2, h3, h4 {
  font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif; }
.stApp { background: var(--page); color: var(--ink); }
/* right-to-left for the content only (the page frame stays as Streamlit expects) */
[data-testid="stMain"], section.main, [data-testid="stSidebarContent"] { direction: rtl; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"], [data-testid="stMain"] p, [data-testid="stMain"] h1,
[data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] h4, [data-testid="stMain"] label { text-align: right; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
/* Streamlit's top bar sits over the page: same colour as the page, content starts below it */
[data-testid="stHeader"] { background: var(--page); }
.block-container { padding-top: 3.6rem !important; padding-bottom: 3rem; max-width: 1240px; }
[data-testid="stMain"] h2 { font-size: 28px !important; font-weight: 700; letter-spacing: 0; margin: 0 !important;
  padding: 0 !important; line-height: 1.25; color: var(--ink); }
[data-testid="stMain"] h4 { font-size: 15px !important; font-weight: 700; color: var(--ink-2); padding: 0 0 4px !important; }
[data-testid="stCaptionContainer"] { color: var(--ink-2); }
.num, .big, .metric .v, .side-sum b { font-variant-numeric: tabular-nums; }
/* ---------- sidebar: the nurse station panel ---------- */
[data-testid="stSidebar"] { background: var(--navy); }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] h3 { font-size: 20px !important; font-weight: 700; padding-bottom: 0 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { opacity: .7; }
[data-testid="stSidebar"] button { background: rgba(255,255,255,.08) !important; border: 1px solid rgba(255,255,255,.14) !important;
  min-height: 48px; border-radius: 12px !important; }
[data-testid="stSidebar"] button:hover { background: rgba(255,255,255,.14) !important; }
.side-sum { border-top: 1px solid rgba(255,255,255,.14); padding: 14px 2px 4px; display:flex; flex-direction:column; gap:12px; font-size:14px; }
.side-sum .head { font-size: 12px; opacity: .65; }
.side-sum b { font-size: 16px; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-inline-end:9px; vertical-align: 1px; }
/* ---------- buttons ---------- */
button[kind="primary"], [data-testid="stBaseButton-primary"] { background: var(--blue) !important; border-color: var(--blue) !important; color: #FFFFFF !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #07466A !important; border-color: #07466A !important; }
button[kind="secondary"], [data-testid="stBaseButton-secondary"] { background: #FFFFFF; border: 1px solid #CFD8E2; color: #24476B; }
.stButton button { border-radius: 10px; min-height: 44px; font-weight: 700; }
/* discharge on a card is a quiet text action, so "open" is the one clear button */
[class*="st-key-dis"] button { background: transparent !important; border: 0 !important; color: var(--ink-2) !important; font-weight: 600; }
[class*="st-key-dis"] button:hover { color: #A11D1D !important; background: #FDEBEA !important; }
/* ---------- bed cards: status shown by a stripe on the right (where a right-to-left reader starts) ---------- */
[class*="st-key-card_"] { background: #FFFFFF; border: 1px solid var(--line) !important; border-radius: 16px;
  padding: 16px 20px 10px 16px; gap: 6px; box-shadow: 0 1px 2px rgba(15,27,45,.04); }
[class*="st-key-card_ok_"]   { box-shadow: inset -5px 0 0 #1F8A4C, 0 1px 2px rgba(15,27,45,.04); }
[class*="st-key-card_warn_"] { box-shadow: inset -5px 0 0 #C98A00, 0 1px 2px rgba(15,27,45,.04); }
[class*="st-key-card_bad_"]  { box-shadow: inset -5px 0 0 #C0392B, 0 6px 18px -10px rgba(192,57,43,.45); border-color: #F3C9C5 !important; }
[class*="st-key-card_done_"] { box-shadow: inset -5px 0 0 #5B7FA6, 0 1px 2px rgba(15,27,45,.04); }
[class*="st-key-empty_"] { background: transparent; border: 1.5px dashed #B5C6D6 !important; border-radius: 16px; padding: 18px 16px; }
[class*="st-key-panel_"] { background: #FFFFFF; border: 1px solid var(--line) !important; border-radius: 16px; padding: 18px 20px; }
.card { display:flex; flex-direction:column; gap:12px; padding-bottom:12px; }
.bed { font-size: 18px; font-weight: 700; }
.big { font-size: 34px; font-weight: 700; line-height: 1; letter-spacing: -.5px; }
.unit { font-size: 13px; color: var(--ink-2); font-weight: 600; }
.reason { font-size: 13.5px; font-weight: 700; padding: 8px 12px; border-radius: 10px; }
/* filter buttons: push to the far left (the end of the row in right-to-left) */
[data-testid="stColumn"]:has(.st-key-filter) [data-testid="stVerticalBlock"],
[data-testid="column"]:has(.st-key-filter) [data-testid="stVerticalBlock"] { align-items:flex-end !important; }
.st-key-filter { width:auto !important; margin-inline-start:auto !important; display:flex; justify-content:flex-end; }
.st-key-filter > div, .st-key-filter [data-testid="stButtonGroup"] { width:auto !important; margin-inline-start:auto; }
/* the segmented buttons draw their joined borders for left-to-right; in right-to-left the outer edges get cut */
.st-key-filter [data-testid="stButtonGroup"], .st-key-filter [role="radiogroup"], .st-key-filter > div > div { direction:ltr; }
.st-key-filter button { min-height: 40px; padding: 0 16px; font-weight: 600; }
.row { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.pill { font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; white-space:nowrap; vertical-align:middle; }
.muted { font-size:12.5px; color: var(--ink-2); }
.box { background:#F5F7FA; border-radius:10px; padding:10px 12px; display:flex; flex-direction:column; gap:6px; font-size:12.5px; }
.track { height:6px; border-radius:999px; background:#E6EBF0; overflow:hidden; }
.track div { height:100%; border-radius:999px; }
.empty { text-align:center; display:flex; flex-direction:column; gap:10px; align-items:center; padding:28px 0 12px; }
.placeholder { border:1.5px dashed #D5DCE4; border-radius:12px; padding:36px 20px; text-align:center; line-height:1.8; }
.banner { border-radius:12px; padding:14px 16px; display:flex; justify-content:space-between; align-items:center; gap:8px; font-weight:700; font-size:16px; }
.metrics { display:flex; flex-direction:column; margin-top:6px; }
.metric { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:14px 2px; border-top:1px solid var(--line); }
.metric:first-child { border-top:0; }
.metric .v { font-size:22px; font-weight:700; margin-top:2px; }
.metric .end { display:flex; flex-direction:column; align-items:flex-end; gap:6px; text-align:left; }
.ltr { direction:ltr; unicode-bidi:isolate; }
/* file pickers: just the button, the file types are enforced anyway */
[data-testid="stFileUploaderDropzoneInstructions"] { display:none; }
[data-testid="stFileUploaderDropzone"] { justify-content:center; padding: 8px; background:#F5F7FA; border:1.5px dashed #C9D4DF; min-width:0; }
[data-testid="stFileUploaderDropzone"] button { width:100%; min-width:0; }
</style>
"""
# Markdown ends an HTML block at the first empty line, so the CSS must have none.
st.markdown("\n".join(l for l in CSS.splitlines() if l.strip()), unsafe_allow_html=True)


# --------------------------------------------------------------- models: load once per server
@st.cache_resource(show_spinner="جاري تجهيز النظام…")
def warmup():
    return backends.warmup()


warmup()


# --------------------------------------------------------------- state
def init_state():
    if "beds" in st.session_state:
        return
    # deep copies: prescription edits change the stored bed, and must not change the shared examples
    st.session_state.beds = {i: ({"kind": "example", "data": copy.deepcopy(EXAMPLE_BEDS[i])} if i in EXAMPLE_BEDS
                                 else {"kind": "free", "data": None}) for i in BED_IDS}
    st.session_state.screen = "board"
    st.session_state.active = 5
    st.session_state.undo = None


init_state()
S = st.session_state


def bed_view(i):
    """Evaluated result for bed i, or None if the bed is free."""
    bed = S.beds[i]
    return None if bed["kind"] == "free" else evaluate_saved(bed["data"])


def open_bed(i):
    S.screen, S.active = "bed", i
    S.undo = None            # the undo offer is short-lived: it ends once the nurse moves on


def go_board():
    S.screen = "board"


def pill(status, text=None):
    t = TONES[status]
    return f'<span class="pill" style="background:{t["bg"]};color:{t["fg"]}">{text or t["label"]}</span>'


DEMO_PILL = '<span class="pill" style="background:#FFF3D6;color:#7A4F00">قيم تجريبية</span>'


# --------------------------------------------------------------- prescription fields
# Streamlit forgets a widget's value when the page that shows it is left, so every time a bed is
# opened its fields are filled again: from the saved prescription (occupied bed) or the defaults.
def rx_keys(i):
    return f"d{i}", f"rx{i}", f"f{i}"


def seed_rx(i):
    saved = S.beds[i]["data"]["inputs"] if S.beds[i]["data"] else DEFAULT_RX
    kd, kr, kf = rx_keys(i)
    S.setdefault(kd, saved["drug"])
    S.setdefault(kr, int(round(saved["rx_mlh"])))
    S.setdefault(kf, int(saved["drop_factor"]))


def sync_rx(i):
    """Occupied bed: apply prescription edits to the stored result at once, without analysing again.
    Rate: mL/h is recomputed from the counted drops. Drug: the label text already read is matched again."""
    d = S.beds[i]["data"]
    if not d:
        return
    inp = d["inputs"]
    kd, kr, kf = rx_keys(i)
    drug, rx, factor = S[kd], S[kr], S[kf]
    if factor != inp["drop_factor"]:
        dpm = float(d["rate"].get("dpm") or 0)
        d["rate"]["ml_h"] = round(dpm * 60 / factor, 1)
        inp["drop_factor"] = factor
    if rx != inp["rx_mlh"]:
        inp["rx_mlh"] = rx
    if drug != inp["drug"]:
        d["label"] = backends.rematch_label(d["label"], drug)
        inp["drug"] = drug


# Done before anything is drawn, so the sidebar counts and the board already use the edited values.
if S.screen == "bed":
    seed_rx(S.active)
    sync_rx(S.active)


# --------------------------------------------------------------- discharge (confirm + undo)
@st.dialog("إخلاء السرير")
def confirm_discharge(i):
    v = bed_view(i)
    st.write(f"إخلاء **سرير {i}**؟ تتوقف مراقبة السرير وتنشال بيانات المريض، ويصير السرير جاهز لمريض جديد.")
    if v and v["status"] != "done":
        left = f'{v["ml_left"]} مل' if v["ml_left"] is not None else (f'{v["pct"]}%' if v["pct"] is not None else "غير معروف")
        st.warning(f"المحلول ما خلص بعد (باقي {left}). تأكدي إن المريض خلص قبل الإخلاء.")
    c1, c2 = st.columns(2)
    if c1.button("إخلاء السرير", type="primary", use_container_width=True):
        S.undo = {"id": i, "bed": S.beds[i]}
        S.beds[i] = {"kind": "free", "data": None}
        S.screen = "board"
        st.rerun()
    if c2.button("إلغاء", use_container_width=True):
        st.rerun()


# --------------------------------------------------------------- sidebar
views = {i: bed_view(i) for i in BED_IDS}
counts = {"ok": 0, "warn": 0, "bad": 0, "done": 0, "free": 0}
for v in views.values():
    counts[v["status"] if v else "free"] += 1
alerts = counts["warn"] + counts["bad"] + counts["done"]
model_status = backends.status()
demo_models = model_status["rate"] == "demo" or model_status["bag"] == "demo"

with st.sidebar:
    st.markdown("### مراقبة المحلول")
    st.caption("محطة التمريض")
    st.button(f"لوحة الأسرّة · {alerts} تنبيه" if alerts else "لوحة الأسرّة", on_click=go_board, use_container_width=True)
    dot = lambda c: f'<span class="dot" style="background:{c}"></span>'
    st.markdown(f"""<div class="side-sum">
      <div class="head">ملخص الوردية</div>
      <div class="row"><span>{dot(TONES['ok']['bar'])}طبيعي</span><b>{counts['ok']}</b></div>
      <div class="row"><span>{dot(TONES['warn']['bar'])}تنبيه</span><b>{counts['warn']}</b></div>
      <div class="row"><span>{dot(TONES['bad']['bar'])}حرج</span><b>{counts['bad']}</b></div>
      <div class="row"><span>{dot(TONES['done']['bar'])}انتهى المحلول</span><b>{counts['done']}</b></div>
      <div class="row"><span>{dot('#7F93A8')}أسرّة فاضية</span><b>{counts['free']}</b></div></div>""", unsafe_allow_html=True)
    if demo_models:
        gaps = backends.missing()
        why = "، ".join((gaps["rate"] if model_status["rate"] == "demo" else []) +
                       (gaps["bag"] if model_status["bag"] == "demo" else []))
        st.caption(f"⚠️ التحليل الجديد يعطي قيم تجريبية. الناقص: {why}")


# --------------------------------------------------------------- board
def render_card(i, v):
    t = TONES[v["status"]]
    pct = v["pct"] if v["pct"] is not None else 0
    bag_t = TONES[v["bag_status"]]
    ml_line = (f'{APPROX}{v["ml_left"]} مل من {v["volume_ml"]} · يخلص بعد {v["time_left"]}' if v["ml_left"] is not None
               else "حجم الكيس ما انقرأ من الملصق")
    if v["status"] == "done":
        ml_line = "خلص"
    label_color = "#14653A" if v["label_ok"] else "#A11D1D"
    demo = f" {DEMO_PILL}" if v["demo"] else ""
    with st.container(border=True, key=f"card_{v['status']}_{i}"):
        st.markdown(f"""
<div class="card">
  <div class="row"><span class="bed">سرير {i}{demo}</span>{pill(v['status'])}</div>
  <div class="row" style="align-items:flex-end">
    <div><span class="big">{v['ml_h']}</span> <span class="unit">مل/س</span>
      <div class="muted" style="margin-top:6px">{v['dpm']:g} قطرة/د · الموصوف {v['rx_mlh']:g}</div></div>
    <span class="ltr">{pill(v['dev_status'], v['dev_text'])}</span>
  </div>
  <div class="box">
    <div class="row"><span class="muted">الموصوف</span><b class="ltr">{v['drug']}</b></div>
    <div class="row"><span class="muted">الملصق</span><b class="ltr" style="color:{label_color}">{v['label_read'] or '—'}</b></div>
  </div>
  <div>
    <div class="row muted" style="margin-bottom:7px"><span>الكيس · <b style="color:#0F1B2D">{v['level_ar']}</b></span><b class="num" style="color:#0F1B2D">{pct}%</b></div>
    <div class="track"><div style="width:{pct}%;background:{bag_t['bar']}"></div></div>
    <div class="muted" style="margin-top:7px">{ml_line}</div>
  </div>
  <div class="reason" style="background:{t['bg']};color:{t['fg']}">{v['reason']}</div>
</div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.button("افتح", key=f"open{i}", on_click=open_bed, args=(i,), use_container_width=True, type="primary")
        if c2.button("إخلاء السرير", key=f"dis{i}", use_container_width=True):
            confirm_discharge(i)


def render_empty(i):
    with st.container(border=True, key=f"empty_{i}"):
        st.markdown(f"""<div class="empty">
      <div style="width:56px;height:56px;border-radius:50%;background:#E3F0F8;color:#0A5C8A;font-size:30px;line-height:56px">+</div>
      <b style="font-size:18px">سرير {i}</b>
      <span class="pill" style="background:#EEF1F4;color:#4A5868">فاضي · جاهز لمريض جديد</span>
      <span class="muted" style="max-width:220px;line-height:1.6">ارفعي الفيديو والصورتين وأدخلي الوصفة</span></div>""",
                    unsafe_allow_html=True)
        st.button("تسجيل مريض", key=f"new{i}", on_click=open_bed, args=(i,), use_container_width=True, type="primary")


def board():
    occupied = sum(1 for v in views.values() if v)
    h1, h2 = st.columns([3, 2], vertical_alignment="center")
    h1.markdown("## لوحة الأسرّة")
    h1.caption(f"6 أسرّة · {occupied} مشغولة")
    # stable option values (the count is only in the shown text), so the choice survives a count change
    options = ["all", "alerts"]
    names = {"all": "الكل", "alerts": f"يحتاج تدخل · {alerts}"}
    if hasattr(st, "segmented_control"):
        # shown left-to-right (see CSS), so reverse the order to keep "الكل" on the right
        flt = h2.segmented_control("تصفية", options[::-1], default="all", format_func=names.get, key="filter",
                                   label_visibility="collapsed")
    else:
        flt = h2.radio("تصفية", options, format_func=names.get, horizontal=True, key="filter",
                       label_visibility="collapsed")
    flt = flt or "all"

    if S.undo:
        u1, u2 = st.columns([4, 1], vertical_alignment="center")
        u1.success(f"تم إخلاء سرير {S.undo['id']} وصار جاهز لمريض جديد")
        if u2.button("تراجع", use_container_width=True):
            if S.beds[S.undo["id"]]["kind"] == "free":          # never overwrite a new patient
                S.beds[S.undo["id"]] = S.undo["bed"]
            S.undo = None
            st.rerun()

    ids = BED_IDS if flt == "all" else [i for i in BED_IDS if views[i] and views[i]["status"] != "ok"]
    if not ids:
        st.info("ما فيه أسرّة تحتاج تدخل الحين.")
    for row_start in range(0, len(ids), 3):          # row by row, so cards in a row line up
        cols = st.columns(3, gap="medium")
        for col, i in zip(cols, ids[row_start:row_start + 3]):
            with col:
                if views[i]:
                    render_card(i, views[i])
                else:
                    render_empty(i)


# --------------------------------------------------------------- bed page
def to_bgr(data):
    """Uploaded photo -> OpenCV BGR image, turned upright (phone photos are often stored sideways
    with an EXIF rotation flag, which would hide the bag from the model and the label from OCR)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def run_analysis(video, bag, label, drug, rx, factor):
    """-> (data, None) or (None, Arabic error message). Never raises."""
    try:
        bag_img, label_img = to_bgr(bag.getvalue()), to_bgr(label.getvalue())
    except Exception:
        return None, "ما قدرنا نفتح الصورة. ارفعي صورة JPG أو PNG."
    suffix = Path(video.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video.getvalue())
        vpath = f.name
    try:
        return backends.analyze_bed(vpath, bag_img, label_img, drug, rx, factor), None
    except FileNotFoundError:
        traceback.print_exc()
        return None, "ما قدرنا نقرأ الفيديو. جرّبي فيديو MP4."
    except Exception:
        traceback.print_exc()
        return None, "صار خطأ أثناء التحليل. جرّبي مرة ثانية."
    finally:
        Path(vpath).unlink(missing_ok=True)


def prescription_fields(i):
    kd, kr, kf = rx_keys(i)
    drugs = KNOWN_DRUGS if S[kd] in KNOWN_DRUGS else KNOWN_DRUGS + [S[kd]]
    st.selectbox("الدواء الموصوف", drugs, key=kd)
    r1, r2 = st.columns(2)
    r1.number_input("السرعة الموصوفة (مل/س)", min_value=1, max_value=1000, step=5, key=kr)
    r2.selectbox("معامل التقطير (قطرة/مل)", FACTORS, key=kf,
                 format_func=lambda f: "60 (أطفال)" if f == 60 else str(f))


def bed_page():
    i = S.active
    v = bed_view(i)

    t1, t2, t3 = st.columns([1.2, 6, 1.6], vertical_alignment="center")
    t1.button("→ رجوع", on_click=go_board, use_container_width=True)
    demo = f" {DEMO_PILL}" if (v and v["demo"]) else ""
    t2.markdown(f"## سرير {i} &nbsp; {pill(v['status'] if v else 'idle')}{demo}", unsafe_allow_html=True)
    if v and t3.button("إخلاء السرير", use_container_width=True):
        confirm_discharge(i)

    left, right = st.columns([1, 1.2], gap="large")

    # ---- inputs
    with left:
        if v:                                         # occupied: the prescription can be corrected
            with st.container(border=True, key=f"panel_rx_{i}"):
                st.markdown("#### الوصفة")
                prescription_fields(i)
                st.caption("أي تعديل على الوصفة يحدّث الحالة مباشرة.")
        else:                                         # free: register a new patient
            with st.container(border=True, key=f"panel_files_{i}"):
                st.markdown("#### الملفات")
                c1, c2, c3 = st.columns(3)
                video = c1.file_uploader("فيديو التقطير", type=["mp4", "mov", "avi", "m4v"], key=f"v{i}")
                bag = c2.file_uploader("صورة الكيس", type=["jpg", "jpeg", "png"], key=f"b{i}")
                label = c3.file_uploader("صورة الملصق", type=["jpg", "jpeg", "png"], key=f"l{i}")
            with st.container(border=True, key=f"panel_rx_{i}"):
                st.markdown("#### الوصفة")
                prescription_fields(i)
                st.caption("حجم الكيس ينقرأ من الملصق تلقائياً.")

            n_files = sum(x is not None for x in (video, bag, label))
            ready = n_files == 3
            label_btn = "حلّل السرير" if ready else f"باقي {3 - n_files} من الملفات"
            if st.button(label_btn, type="primary", disabled=not ready, use_container_width=True):
                kd, kr, kf = rx_keys(i)
                with st.spinner("جاري التحليل…"):
                    data, err = run_analysis(video, bag, label, str(S[kd]), int(S[kr]), int(S[kf]))
                if err:
                    st.error(err)
                else:
                    S.beds[i] = {"kind": "live", "data": data}
                    if S.undo and S.undo["id"] == i:
                        S.undo = None
                    st.rerun()

    # ---- result
    with right, st.container(border=True, key=f"panel_result_{i}"):
        st.markdown("#### النتيجة")
        if not v:
            st.markdown('<div class="placeholder"><span class="muted">النتيجة تظهر هنا بعد التحليل: السرعة، المتبقي في الكيس، '
                        'ومطابقة الملصق</span></div>', unsafe_allow_html=True)
        else:
            result_panel(v)



def result_panel(v):
    head = TONES[v["status"]]
    st.markdown(f'<div class="banner" style="background:{head["bg"]};color:{head["fg"]};border:1px solid {head["border"]}">'
                f'<span>{head["label"]}</span><span style="font-weight:600">{v["reason"]}</span></div>',
                unsafe_allow_html=True)
    left_v = f'{APPROX}{v["ml_left"]} مل' if v["ml_left"] is not None else (f'{v["pct"]}%' if v["pct"] is not None else "—")
    tag = "مطابق" if v["label_ok"] else "غير مطابق"
    # one row per check: name and value on the right, verdict and detail on the left
    st.markdown(f"""<div class="metrics">
      <div class="metric"><div><div class="muted">السرعة</div><div class="v">{v['ml_h']} <span class="unit">مل/س</span></div></div>
        <div class="end"><span class="ltr">{pill(v['dev_status'], v['dev_text'])}</span>
        <div class="muted">{v['dpm']:g} قطرة/د · الموصوف {v['rx_mlh']:g}</div></div></div>
      <div class="metric"><div><div class="muted">المتبقي</div><div class="v">{left_v}</div></div>
        <div class="end">{pill(v['bag_status'], v['level_ar'])}
        <div class="muted">من {v['volume_ml'] or '؟'} مل · يخلص بعد {v['time_left']}</div></div></div>
      <div class="metric"><div><div class="muted">الملصق</div><div class="v ltr">{v['label_read'] or '—'}</div></div>
        <div class="end">{pill('ok' if v['label_ok'] else 'bad', tag)}
        <div class="muted">الموصوف: <span class="ltr">{v['drug']}</span></div></div></div>
    </div>""", unsafe_allow_html=True)


# --------------------------------------------------------------- router
if S.screen == "bed":
    bed_page()
else:
    board()
