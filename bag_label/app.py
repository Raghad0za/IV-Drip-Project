"""
IV monitoring demo: nurse dashboard (Arabic, RTL).

    streamlit run app/app.py

Board of 6 beds -> open a bed -> upload drip video + bag photo + label photo + prescription
-> analyse with Jana's measure_rate and Hasan's bag_level / check_label -> result on the board.
The nurse sees only what she acts on. Charts and raw outputs are behind the sidebar switch
"وضع العرض التقني" (off by default).
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
from iv_logic import EXAMPLE_BEDS, TONES, evaluate_saved

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
html, body, .stApp, button, input, textarea, select, p, span, div, label, h1, h2, h3, h4 {
  font-family: 'IBM Plex Sans Arabic', system-ui, sans-serif; }
.stApp { background: #F3F5F8; }
/* right-to-left for the content only (the page frame stays as Streamlit expects) */
[data-testid="stMain"], section.main, [data-testid="stSidebarContent"] { direction: rtl; }
[data-testid="stMain"] p, [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] label { text-align: right; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
h2 { font-size: 26px !important; padding-bottom: 0 !important; }
h4 { font-size: 16px !important; }
/* sidebar = dark navy panel of the mockup */
[data-testid="stSidebar"] { background: #0E2A3F; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] button { background: #1D4A6B !important; border: 0 !important; min-height: 48px; }
/* buttons */
button[kind="primary"], [data-testid="stBaseButton-primary"] { background: #0A5C8A !important; border-color: #0A5C8A !important; color: #FFFFFF !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #07466A !important; }
button[kind="secondary"], [data-testid="stBaseButton-secondary"] { background: #FFFFFF; border: 1px solid #D5DCE4; color: #24476B; }
.stButton button { border-radius: 10px; min-height: 44px; font-weight: 700; }
/* bed cards: a bordered container whose key starts with card_<status>_ */
[class*="st-key-card_"] { background: #FFFFFF; border: 1px solid #E2E7ED !important; border-radius: 18px;
  padding: 14px 16px 12px; box-shadow: 0 1px 2px rgba(15,27,45,.05), 0 12px 24px -18px rgba(15,27,45,.3); gap: 10px; }
[class*="st-key-card_ok_"]   { border-top: 5px solid #1F8A4C !important; }
[class*="st-key-card_warn_"] { border-top: 5px solid #C98A00 !important; }
[class*="st-key-card_bad_"]  { border-top: 5px solid #C0392B !important; }
[class*="st-key-card_done_"] { border-top: 5px solid #5B7FA6 !important; background: #FBFCFE; }
[class*="st-key-empty_"] { background: #F8FAFC; border: 1.5px dashed #9AB3C8 !important; border-radius: 18px; padding: 18px 16px; }
[class*="st-key-panel_"] { background: #FFFFFF; border: 1px solid #E2E7ED !important; border-radius: 18px; padding: 16px; }
.card { display:flex; flex-direction:column; gap:10px; padding-bottom:14px; }
/* filter buttons: push to the far left (the end of the row in right-to-left) */
[data-testid="stColumn"]:has(.st-key-filter) [data-testid="stVerticalBlock"],
[data-testid="column"]:has(.st-key-filter) [data-testid="stVerticalBlock"] { align-items:flex-end !important; }
.st-key-filter { width:auto !important; margin-inline-start:auto !important; display:flex; justify-content:flex-end; }
.st-key-filter > div, .st-key-filter [data-testid="stButtonGroup"] { width:auto !important; margin-inline-start:auto; }
/* the segmented buttons draw their joined borders for left-to-right; in right-to-left the outer edges get cut */
.st-key-filter [data-testid="stButtonGroup"], .st-key-filter [role="radiogroup"], .st-key-filter > div > div { direction:ltr; }
.row { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.pill { font-size:12px; font-weight:700; padding:3px 10px; border-radius:999px; white-space:nowrap; }
.big { font-size:30px; font-weight:700; line-height:1; }
.muted { font-size:12px; color:#4A5868; }
.box { background:#F5F7F9; border-radius:12px; padding:9px 12px; display:flex; flex-direction:column; gap:5px; font-size:12px; }
.track { height:8px; border-radius:999px; background:#E9EDF2; overflow:hidden; }
.track div { height:100%; border-radius:999px; }
.empty { text-align:center; display:flex; flex-direction:column; gap:8px; align-items:center; padding:10px 0 4px; }
.placeholder { border:1.5px dashed #D5DCE4; border-radius:14px; padding:28px 16px; text-align:center; }
.banner { border-radius:12px; padding:11px 14px; display:flex; justify-content:space-between; gap:8px; font-weight:700; }
.metric { background:#FFFFFF; border:1px solid #E2E7ED; border-radius:12px; padding:11px 12px; display:flex; flex-direction:column; gap:4px; }
.metric .v { font-size:18px; font-weight:700; }
.ltr { direction:ltr; unicode-bidi:isolate; }
.side-sum { background:#143650; border-radius:14px; padding:14px; display:flex; flex-direction:column; gap:8px; font-size:14px; }
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
    st.session_state.last_error = None


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
    st.markdown("### 💧 مراقبة المحلول")
    st.caption("محطة التمريض")
    st.button(f"لوحة الأسرّة · {alerts} تنبيه" if alerts else "لوحة الأسرّة", on_click=go_board, use_container_width=True)
    st.markdown(f"""<div class="side-sum">
      <div style="opacity:.75;font-size:12px">ملخص الوردية</div>
      <div class="row"><span>🟢 طبيعي</span><b>{counts['ok']}</b></div>
      <div class="row"><span>🟡 تنبيه</span><b>{counts['warn']}</b></div>
      <div class="row"><span>🔴 حرج</span><b>{counts['bad']}</b></div>
      <div class="row"><span>🔵 انتهى المحلول</span><b>{counts['done']}</b></div>
      <div class="row"><span>أسرّة فاضية</span><b>{counts['free']}</b></div></div>""", unsafe_allow_html=True)
    if demo_models:
        st.caption("⚠️ بعض الموديلات غير متوفرة على هذا الجهاز: التحليل الجديد يعطي قيم تجريبية.")
    st.markdown("---")
    tech = st.toggle("وضع العرض التقني", key="tech")
    if tech:
        st.caption({"yolo": "✅ القطرات: YOLO (drops.pt) + طريقة الحركة احتياط",
                    "motion": "⚠️ القطرات: طريقة الحركة (YOLO غير متوفر)",
                    "demo": "⚠️ القطرات: قيم تجريبية"}[model_status["rate"]])
        st.caption("✅ الكيس والملصق: الموديل الحقيقي" if model_status["bag"] == "real" else "⚠️ الكيس والملصق: قيم تجريبية")
        st.caption(f"مدة الفيديو المحلَّلة: أول {backends.MAX_SECONDS:g} ثانية")
        for part, err in {**model_status["disabled"], "rate.py": model_status["rate_error"],
                          "iv_monitor.py": model_status["bag_error"]}.items():
            if err:
                st.caption(f"{part}: {err}")


# --------------------------------------------------------------- board
def render_card(i, v):
    t = TONES[v["status"]]
    pct = v["pct"] if v["pct"] is not None else 0
    bag_t = TONES[v["bag_status"]]
    ml_line = (f'~{v["ml_left"]} مل من {v["volume_ml"]} · يخلص بعد {v["time_left"]}' if v["ml_left"] is not None
               else "حجم الكيس ما انقرأ من الملصق")
    if v["status"] == "done":
        ml_line = "خلص"
    label_color = "#14653A" if v["label_ok"] else "#A11D1D"
    demo = f" {DEMO_PILL}" if v["demo"] else ""
    with st.container(border=True, key=f"card_{v['status']}_{i}"):
        st.markdown(f"""
<div class="card">
  <div class="row"><b style="font-size:17px">سرير {i}{demo}</b>{pill(v['status'])}</div>
  <div class="row" style="align-items:flex-end">
    <div><span class="big">{v['ml_h']}</span> <span class="muted">مل/س</span>
      <div class="muted" style="margin-top:5px">{v['dpm']:g} قطرة/د · الموصوف {v['rx_mlh']:g}</div></div>
    <span class="ltr">{pill(v['dev_status'], v['dev_text'])}</span>
  </div>
  <div class="box">
    <div class="row"><span class="muted">الموصوف</span><b class="ltr">{v['drug']}</b></div>
    <div class="row"><span class="muted">الملصق</span><b class="ltr" style="color:{label_color}">{v['label_read'] or '—'}</b></div>
  </div>
  <div class="row muted"><span>الكيس · <b style="color:#0F1B2D">{v['level_ar']}</b></span><b style="color:#0F1B2D">{pct}%</b></div>
  <div class="track"><div style="width:{pct}%;background:{bag_t['bar']}"></div></div>
  <div class="muted">{ml_line}</div>
  <div style="font-size:13px;font-weight:600;color:{t['fg']};border-top:1px solid #EEF1F4;padding-top:10px">{v['reason']}</div>
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
        S.last_error = traceback.format_exc()
        return None, "ما قدرنا نقرأ الفيديو. جرّبي فيديو MP4."
    except Exception:
        S.last_error = traceback.format_exc()
        traceback.print_exc()
        return None, "صار خطأ أثناء التحليل. جرّبي مرة ثانية."
    finally:
        Path(vpath).unlink(missing_ok=True)


def drop_chart(rate):
    """Change / drop-edge signal with the counted drops, like rate.plot_debug (technical mode only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rate.get("method") == "motion" and rate.get("motion_signal"):
        fig, ax = plt.subplots(figsize=(7, 2.2))
        sig = rate["motion_signal"]
        ax.plot([p[0] for p in sig], [p[1] for p in sig], lw=1, color="#2a78d6")
        if rate.get("threshold") is not None:
            ax.axhline(rate["threshold"], color="#8a8a85", lw=1, ls="--")
        ax.set_ylabel("change")
    elif rate.get("signal"):
        fig, ax = plt.subplots(figsize=(7, 2.2))
        pts = [p for p in rate["signal"] if p]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=3, color="#2a78d6")
        ax.invert_yaxis()
        ax.set_ylabel("drop bottom (px)")
    else:
        return None
    for t in rate.get("events", []):
        ax.axvline(t, color="#e5484d", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_title(f"{rate.get('drops', 0)} drops · {rate.get('dpm', 0)} dpm · {rate.get('method')}", fontsize=9)
    fig.tight_layout()
    return fig


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

    t1, t2, t3 = st.columns([1, 5, 2], vertical_alignment="center")
    t1.button("→ رجوع", on_click=go_board, use_container_width=True)
    demo = f" {DEMO_PILL}" if (v and v["demo"]) else ""
    t2.markdown(f"## سرير {i} &nbsp; {pill(v['status'] if v else 'idle')}{demo}", unsafe_allow_html=True)
    if v and t3.button("إخلاء السرير", use_container_width=True):
        confirm_discharge(i)

    left, right = st.columns([1, 1.05], gap="large")

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

    if S.get("tech"):
        tech_panel(i)


def result_panel(v):
    head = TONES[v["status"]]
    st.markdown(f'<div class="banner" style="background:{head["bg"]};color:{head["fg"]};border:1px solid {head["border"]}">'
                f'<span>{head["label"]}</span><span style="font-weight:600">{v["reason"]}</span></div>',
                unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric"><div class="row"><span class="muted">السرعة</span>
        <span class="ltr">{pill(v['dev_status'], v['dev_text'])}</span></div>
        <div class="v">{v['ml_h']} مل/س</div>
        <div class="muted">{v['dpm']:g} قطرة/د · الموصوف {v['rx_mlh']:g}</div></div>""", unsafe_allow_html=True)
    left_v = f'~{v["ml_left"]} مل' if v["ml_left"] is not None else (f'{v["pct"]}%' if v["pct"] is not None else "—")
    m2.markdown(f"""<div class="metric"><div class="row"><span class="muted">المتبقي</span>
        {pill(v['bag_status'], v['level_ar'])}</div>
        <div class="v">{left_v}</div>
        <div class="muted">من {v['volume_ml'] or '؟'} مل · يخلص بعد {v['time_left']}</div></div>""", unsafe_allow_html=True)
    tag = "مطابق" if v["label_ok"] else "غير مطابق"
    m3.markdown(f"""<div class="metric"><div class="row"><span class="muted">الملصق</span>
        {pill('ok' if v['label_ok'] else 'bad', tag)}</div>
        <div class="v ltr">{v['label_read'] or '—'}</div>
        <div class="muted">الموصوف: <span class="ltr">{v['drug']}</span></div></div>""", unsafe_allow_html=True)


def tech_panel(i):
    """For the evaluator / the team: how the numbers were produced. Hidden from the nurse by default."""
    data = S.beds[i]["data"]
    st.markdown("---")
    st.markdown("#### تفاصيل تقنية")
    if data:
        fig = drop_chart(data["rate"])
        if fig is not None:
            import matplotlib.pyplot as plt
            st.pyplot(fig)
            plt.close(fig)
            st.caption("كل خط أحمر = قطرة انعدّت.")
        st.json({"rate": {k: data["rate"].get(k) for k in ("dpm", "ml_h", "drops", "status", "method", "yolo_probe",
                                                             "detection_rate", "rate_source", "events")},
                 "bag": data["bag"], "label": data["label"]},
                expanded=False)
    if S.last_error:
        with st.expander("آخر خطأ"):
            st.code(S.last_error)


# --------------------------------------------------------------- router
if S.screen == "bed":
    bed_page()
else:
    board()
