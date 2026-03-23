# ⚡ VisionForge — Model Training Studio

Train custom YOLOv8 object detection models from your Roboflow dataset using a single Streamlit file. No Flask, no backend server, no extra processes — just one command.

---

## 📁 Project Structure

```
visionforge/
├── app.py            # Everything — UI + training logic
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`

> `roboflow` and `ultralytics` are installed automatically when training starts — no need to add them manually.

---

## 🖥️ How the UI Works

The app is divided into 4 steps. Complete them top to bottom.

---

### Step 1 · Roboflow Dataset

Paste your Roboflow download snippet into the text box:

```python
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("my-workspace").project("my-project")
version = project.version(1)
dataset = version.download("yolov8")
```

Click **Parse Snippet**. The app uses regex to extract your API key, workspace, project name, and version number and displays them as read-only fields.

> ⚠️ Start Training is disabled until a snippet is successfully parsed.

---

### Step 2 · Hyperparameters

| Setting | Options | Default |
|---|---|---|
| **Epochs** | 1 – 500 | 25 |
| **Image Size** | 640px / 800px / 1024px | 640px |
| **Model** | YOLOv8n / YOLOv8s / YOLOv8m | YOLOv8n |

**▶ Start Training** — launches training in a background thread. Disabled until Step 1 is complete.

**⬛ Cancel Run** — sends `SIGINT` to the training subprocess to stop it cleanly. Only active while training is running.

---

### Step 3 · Live Training Log

Once training starts the page auto-refreshes every second and shows:

- **Status badge** — `🟢 Training running…` or `✅ Training complete!`
- **Progress bar** — updates as each epoch completes, parsed directly from the log output
- **Log window** — streams every line from the training subprocess in real time (dataset download, GPU/CPU detection, per-epoch metrics, errors). Shows the last 80 lines.
- **Best weights path** — displayed after training finishes, e.g. `runs/detect/train/weights/best.pt`

#### How the live log works (no Flask needed)

Training runs inside a `subprocess.Popen` spawned by a `threading.Thread`. The thread feeds every log line into a `queue.Queue`. On each page refresh, the main Streamlit script drains the queue and appends lines to `st.session_state.log_lines`. The thread never touches `st.session_state` directly — this avoids Streamlit's `missing ScriptRunContext` warning.

---

### Step 4 · Training Results

After training finishes, `results.png` is read from the most recent `runs/detect/train*/` folder and displayed — showing loss curves, precision, recall, and mAP across epochs.

---

## 📦 Requirements

```
streamlit
```

That's it. Everything else is auto-installed at training time.

---

## 💡 Tips

- **YOLOv8n** trains fastest — good for testing your dataset and config before committing to a long run
- **YOLOv8m** gives the best accuracy but needs a GPU for reasonable training time
- Larger image sizes (800px, 1024px) help detect small objects but slow down training
- Trained weights are saved to `runs/detect/train*/weights/` — `best.pt` is the checkpoint with the highest validation mAP
- If you cancel and restart, a new `train2/`, `train3/` etc. folder is created — previous runs are not overwritten
