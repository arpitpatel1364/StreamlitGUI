"""
VisionForge — Standalone Streamlit App
No Flask required. Run:  streamlit run app.py
"""

import os
import re
import glob
import queue
import signal
import threading
import subprocess
import streamlit as st

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VisionForge", page_icon="⚡", layout="centered")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("parsed",           None),
    ("training_started", False),
    ("training_done",    False),
    ("log_lines",        []),
    ("weights_path",     ""),
    ("proc_holder",      [None]),   # list so thread can mutate without touching session_state
    ("log_queue",        queue.Queue()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_snippet(snippet: str):
    api_key   = re.search(r'api_key\s*=\s*["\']([^"\']+)["\']', snippet)
    workspace = re.search(r'\.workspace\(["\']([^"\']+)["\']\)', snippet)
    project   = re.search(r'\.project\(["\']([^"\']+)["\']\)', snippet)
    version   = re.search(r'\.version\((\d+)\)', snippet)
    missing = [n for n, v in [("api_key", api_key), ("workspace", workspace),
                               ("project name", project), ("version", version)] if not v]
    if missing:
        return None, f"Could not parse: {', '.join(missing)}"
    return {
        "api_key":      api_key.group(1),
        "workspace":    workspace.group(1),
        "project_name": project.group(1),
        "version_num":  int(version.group(1)),
    }, None


def find_best_weights():
    candidates = sorted(glob.glob("VisionForge/runs/detect/train*/weights/best.pt"),
                        key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def find_results_image():
    candidates = sorted(glob.glob("VisionForge/runs/detect/train*/results.png"),
                        key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def build_training_script(cfg: dict) -> str:
    return f"""
import subprocess, sys
for pkg in ["roboflow", "ultralytics"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

import torch
if torch.cuda.is_available():
    device, amp = "0", True
    print(f"✅ GPU: {{torch.cuda.get_device_name(0)}}")
else:
    device, amp = "cpu", False
    print("ℹ️  No GPU detected — training on CPU")

from roboflow import Roboflow
from ultralytics import YOLO

print("📦 Connecting to Roboflow…")
rf      = Roboflow(api_key="{cfg['api_key']}")
project = rf.workspace("{cfg['workspace']}").project("{cfg['project_name']}")
version = project.version({cfg['version_num']})
dataset = version.download("yolov8")
print(f"✅ Dataset ready: {{dataset.location}}")

print("🏋️  Starting YOLOv8 training…")
model = YOLO("{cfg['model_variant']}")
model.train(
    data=f"{{dataset.location}}/data.yaml",
    epochs={cfg['epochs']},
    imgsz={cfg['image_size']},
    device=device,
    amp=amp,
    verbose=True,
)
print("✅ Training finished.")
"""


def run_training_thread(cfg: dict, log_q: queue.Queue, proc_holder: list):
    """Background thread — NEVER touches st.session_state.
    All communication back to the main script goes through log_q.
    proc_holder is a one-element list so the main thread can cancel."""
    script = build_training_script(cfg)
    try:
        log_q.put("🔍 Downloading dataset from Roboflow…")
        proc = subprocess.Popen(
            ["python", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        proc_holder[0] = proc          # share handle for cancel — no session_state

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log_q.put(line)

        proc.wait()

        if proc.returncode == 0:
            w = find_best_weights()
            log_q.put("✅ Training complete!")
            if w:
                log_q.put(f"🚀 Best weights: {w}")
                log_q.put(f"__WEIGHTS__{w}")   # structured sentinel for weights path
        else:
            log_q.put(f"❌ Process exited with code {proc.returncode}")

    except Exception as exc:
        log_q.put(f"❌ Exception: {exc}")
    finally:
        proc_holder[0] = None
        log_q.put("__DONE__")



st.title("⚡ VisionForge — Model Training Studio")
st.caption("Roboflow · Ultralytics YOLOv8  — Streamlit")
st.divider()

st.subheader("Step 1 · Roboflow Dataset")

snippet = st.text_area(
    "Paste your Roboflow code snippet",
    height=120,
    placeholder=(
        'rf = Roboflow(api_key="YOUR_KEY")\n'
        'project = rf.workspace("workspace").project("project-name")\n'
        'version = project.version(1)\n'
        'dataset = version.download("yolov8")'
    ),
)

if st.button("🔍 Parse Snippet", use_container_width=True):
    if not snippet.strip():
        st.error("Paste your Roboflow snippet first.")
    else:
        result, err = parse_snippet(snippet)
        if err:
            st.error(err)
        else:
            st.session_state.parsed = result
            st.success(
                f"✅  **{result['project_name']}** · v{result['version_num']} · {result['workspace']}"
            )

if st.session_state.parsed:
    p = st.session_state.parsed
    c1, c2 = st.columns(2)
    c1.text_input("API Key",   value=p["api_key"],          disabled=True)
    c1.text_input("Project",   value=p["project_name"],     disabled=True)
    c2.text_input("Workspace", value=p["workspace"],        disabled=True)
    c2.text_input("Version",   value=str(p["version_num"]), disabled=True)

st.divider()


st.subheader("Step 2 · Hyperparameters")

col1, col2, col3 = st.columns(3)
epochs        = col1.number_input("Epochs", min_value=1, max_value=500, value=25)
image_size    = col2.selectbox("Image Size", [640, 800, 1024],
                               format_func=lambda x: f"{x}px")
model_variant = col3.selectbox("Model",
                               ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                               format_func=lambda x: x.replace(".pt", "").upper())

b1, b2 = st.columns(2)
start_clicked  = b1.button(
    "▶ Start Training",
    disabled=(st.session_state.parsed is None or st.session_state.training_started),
    type="primary",
    use_container_width=True,
)
cancel_clicked = b2.button(
    "⬛ Cancel Run",
    disabled=not st.session_state.training_started,
    use_container_width=True,
)

# after start 

if start_clicked and st.session_state.parsed:
    cfg = {
        **st.session_state.parsed,
        "epochs":        str(int(epochs)),
        "image_size":    str(image_size),
        "model_variant": model_variant,
    }
    st.session_state.training_started = True
    st.session_state.training_done    = False
    st.session_state.log_lines        = []
    st.session_state.weights_path     = ""
    st.session_state.log_queue        = queue.Queue()
    st.session_state.proc_holder      = [None]

    threading.Thread(
        target=run_training_thread,
        args=(cfg, st.session_state.log_queue, st.session_state.proc_holder),
        daemon=True,
    ).start()
    st.rerun()

if cancel_clicked and st.session_state.proc_holder[0] is not None:
    try:
        os.killpg(os.getpgid(st.session_state.proc_holder[0].pid), signal.SIGINT)
        st.session_state.training_started = False
        st.session_state.log_queue.put("Training cancelled by user.")
        st.session_state.log_queue.put("__DONE__")
        st.warning("⚠ Cancel signal sent.")
    except Exception as e:
        st.error(f"Cancel failed: {e}")

st.divider()

st.subheader("Step 3 · Live Training Log")

if not st.session_state.training_started and not st.session_state.training_done:
    st.caption("── waiting for training to start ──")

else:
    status_box   = st.empty()
    progress_bar = st.progress(0)
    log_box      = st.empty()

    # Drain queue into log_lines
    log_q = st.session_state.log_queue
    done  = False
    while True:
        try:
            line = log_q.get_nowait()
            if line == "__DONE__":
                done = True
                break
            elif line.startswith("__WEIGHTS__"):
                st.session_state.weights_path = line[len("__WEIGHTS__"):]
            else:
                st.session_state.log_lines.append(line)
        except queue.Empty:
            break

    # Parse latest epoch for progress
    pct = 0
    for l in reversed(st.session_state.log_lines):
        m = re.search(r"[Ee]poch\s+(\d+)\s*/\s*(\d+)", l)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            pct = int(cur / tot * 100)
            break

    # Render status + log
    if st.session_state.training_done or done:
        status_box.success("Training complete!")
        progress_bar.progress(100)
    else:
        status_box.info("Training running…")
        progress_bar.progress(pct)

    log_box.code(
        "\n".join(st.session_state.log_lines[-80:]) or "Starting…",
        language=None,
    )

    # Auto-refresh every second while running
    if st.session_state.training_started and not done:
        import time
        time.sleep(1)
        st.rerun()

    # Finalise on __DONE__
    if done and not st.session_state.training_done:
        st.session_state.training_done    = True
        st.session_state.training_started = False

    if st.session_state.weights_path:
        st.info(f"**Best weights →** `{st.session_state.weights_path}`")

st.divider()

if st.session_state.training_done:
    st.subheader("Step 4 · Training Results")
    img_path = find_results_image()
    if img_path and os.path.exists(img_path):
        st.image(img_path, caption="Metrics & curves", use_container_width=True)
    else:
        st.warning("Results image not available yet.")