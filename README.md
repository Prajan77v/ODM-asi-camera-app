# 🚨 AI Smart Surveillance System v4.0

A futuristic real-time AI surveillance platform powered by:

- 🧠 YOLOv8 Object Detection
- 👤 Face Recognition
- 📷 Multi-Camera Monitoring
- ⚡ RTX GPU Acceleration
- 📲 Telegram Alerts
- 🧵 Async Processing
- 🌌 Cyberpunk UI
- 📊 Structured Logging
- 🔥 Ultra Smooth Real-Time Rendering

Designed for:
- Home surveillance
- Office security
- AI monitoring systems
- Research projects
- Smart security automation

---

# ✨ Features

## 🎯 AI Detection
- Real-time YOLOv8 object detection
- Person tracking
- Intruder detection
- Face recognition
- Multi-object monitoring

---

## 📲 Telegram Notifications

Beautiful real-time alerts with emojis.

Example:

🚨 INTRUDER DETECTED  
📷 Camera: Laptop Cam  
👤 Unknown Person  
🎯 Confidence: 94%  
⏰ 11:39 PM

Supports:
- Image alerts
- Event alerts
- Object alerts
- Person entry/exit alerts

---

## 📷 Multi Camera Support

Supports:
- Laptop camera
- USB cameras
- IP cameras
- DroidCam
- RTSP streams
- Android IP Webcam

---

## ⚡ High Performance

Optimized for:
- RTX GPUs
- CUDA acceleration
- Multi-threading
- Async queues
- Low latency rendering

Features:
- Smooth UI
- Adaptive rendering
- Frame skipping
- Non-blocking notifications
- Fast reconnect system

---

## 🌌 Modern Cyberpunk UI

Includes:
- AI dashboard
- Live event feed
- Smooth overlays
- Responsive camera grid
- Camera health monitoring
- FPS monitoring
- Futuristic surveillance aesthetic

---

# 📂 Project Structure

```text
AI-Surveillance/
│
├── surveillance.py
├── requirements.txt
├── alarm.wav
│
├── logs/
│   ├── app.log
│   ├── errors.log
│   ├── events.log
│   ├── events.jsonl
│   └── events_export.csv
│
├── faces/
│   ├── known/
│   ├── unknown/
│   └── captured/
│
└── README.md
```

---

# 🖥️ Installation Guide

# 🐧 Linux Installation (Arch / Ubuntu)

## 1️⃣ Install Python

### Arch Linux

```bash
sudo pacman -S python python-pip
```

### Ubuntu

```bash
sudo apt update
sudo apt install python3 python3-pip -y
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

### Linux

```bash
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

OR manually:

```bash
pip install ultralytics opencv-python face_recognition numpy requests psutil
```

---

## 4️⃣ Install CUDA (Optional but Recommended)

For RTX GPU acceleration install:
- NVIDIA drivers
- CUDA Toolkit
- cuDNN

Verify CUDA:

```bash
nvidia-smi
```

---

## 5️⃣ Run the System

```bash
python surveillance.py
```

---

# 🪟 Windows Installation Guide

## 1️⃣ Install Python

Download:
https://www.python.org/downloads/windows/

IMPORTANT:
During installation ENABLE:

✅ Add Python to PATH

---

## 2️⃣ Install Visual Studio Build Tools

Needed for `face_recognition` and `dlib`.

Download:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

During install select:
- Desktop development with C++
- MSVC compiler
- Windows SDK

---

## 3️⃣ Install Git (Optional)

https://git-scm.com/download/win

---

## 4️⃣ Open CMD or PowerShell

Navigate to project:

```powershell
cd Desktop\AI-Surveillance
```

---

## 5️⃣ Create Virtual Environment

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\activate
```

---

## 6️⃣ Install Dependencies

```powershell
pip install -r requirements.txt
```

OR:

```powershell
pip install ultralytics opencv-python face_recognition numpy requests psutil
```

---

## 7️⃣ Install PyTorch CUDA (RTX GPU)

Visit:
https://pytorch.org/get-started/locally/

Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 8️⃣ Run the Program

```powershell
python surveillance.py
```

---

# 📲 Telegram Bot Setup

## 1️⃣ Create Bot

Open Telegram:
https://t.me/BotFather

Command:

```text
/newbot
```

Copy:
- Bot Token

---

## 2️⃣ Get Chat ID

Send a message to your bot.

Open:

```text
https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
```

Copy:
- chat ID

---

## 3️⃣ Add To Config

Inside code:

```python
BOT_TOKEN = "YOUR_TOKEN"
CHAT_ID   = "YOUR_CHAT_ID"
```

---

# 📷 IP Camera Setup

## Android Phone Camera

Install:

### IP Webcam
https://play.google.com/store/apps/details?id=com.pas.webcam

OR

### DroidCam
https://www.dev47apps.com/

Use URL:

```python
"http://192.168.x.x:8080/video"
```

Example:

```python
CAMERA_CONFIGS = [
    {"source": 0, "name": "Laptop Cam", "enabled": True},
    {"source": "http://192.168.1.100:8080/video", "name": "Phone Cam", "enabled": True},
]
```

---

# ⌨️ Controls

| Key | Action |
|---|---|
| G | Grid View |
| F | Focus Mode |
| TAB | Switch Camera |
| 1-9 | Select Camera |
| L | Toggle Left Panel |
| R | Toggle Event Panel |
| B | Toggle Footer |
| S | Send Evidence |
| X | Export CSV Logs |
| Q | Quit |

---

# 📊 Logging System

Generated logs:

| File | Purpose |
|---|---|
| app.log | General logs |
| errors.log | Error logs |
| events.log | Human readable events |
| events.jsonl | Structured JSON logs |
| events_export.csv | CSV export |

---

# 🧠 Face Recognition

Add known faces here:

```text
faces/known/
```

Example:

```text
faces/known/Prajan.jpg
```

---

# ⚡ Performance Tips

## Best Settings For RTX GPUs

```python
MODEL_NAME = "yolov8n.pt"
DEVICE = "cuda"
PROCESS_EVERY_N = 3
```

---

## For Ultra Smooth FPS

Reduce resolution:

```python
FRAME_W = 640
FRAME_H = 360
```

---

# 🛠 Troubleshooting

## Camera Not Working

Linux:

```bash
v4l2-ctl --list-devices
```

Windows:
- Check Camera permissions
- Close apps using webcam

---

## Telegram Not Sending

Check:
- Bot token
- Chat ID
- Internet connection

Test:

```python
requests.get("https://api.telegram.org/botTOKEN/getMe")
```

---

## CUDA Not Working

Check:

```bash
nvidia-smi
```

If CUDA unavailable:
System falls back to CPU automatically.

---

# 🚀 Future Plans

- Web dashboard
- Mobile app
- Voice alerts
- AI anomaly detection
- Heatmaps
- License plate recognition
- Cloud storage
- Remote control panel

---

# 👨‍💻 Author

Prajan  
AI + Computer Vision + Surveillance Systems

---

# ⭐ GitHub Setup

## Create Repository

https://github.com

---

## Push Project

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin YOUR_REPO_LINK
git push -u origin main
```

---

# 📜 License

MIT License

Free to use and modify.
