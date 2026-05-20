# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║     AI SMART SURVEILLANCE  v6.0  —  ZERO-LAG · ZERO-DUPLICATE            ║
# ║  Single/Multi-cam · Adaptive LOW-END · Cyberpunk UI · Telegram dedup     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# 

from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import logging.handlers
import math
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX   = platform.system() == "Linux"

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed")

try:
    import face_recognition
    FACE_RECOG_AVAILABLE = True
except ImportError:
    FACE_RECOG_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_DEVICE    = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE    = "CPU"

if IS_WINDOWS:
    try:
        import winsound
        WINSOUND_AVAILABLE = True
    except ImportError:
        WINSOUND_AVAILABLE = False
else:
    WINSOUND_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# HARDWARE PROFILE AUTO-DETECT
# ══════════════════════════════════════════════════════════════════════════════

def _detect_profile() -> str:
    if CUDA_AVAILABLE:
        return "HIGH"
    cores  = os.cpu_count() or 2
    ram_gb = (psutil.virtual_memory().total / (1024**3)) if PSUTIL_AVAILABLE else 4.0
    if cores >= 6 and ram_gb >= 8:
        return "MEDIUM"
    return "LOW"

HW_PROFILE = _detect_profile()
print(f"\n[INIT] Profile={HW_PROFILE}  CUDA={CUDA_AVAILABLE}  GPU={CUDA_DEVICE}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    BOT_TOKEN      = "8938780809:AAHzpgv_fbfbmXJ9x_ui44LY83CWnTWfKPo"
    CHAT_ID        = "8076971661"
    TG_TIMEOUT     = 10
    TG_MAX_RETRIES = 3
    TG_RETRY_DELAY = 2.0
    TG_QUEUE_SIZE  = 128

    # [FIX-1] Per-event cooldown AND global message-hash dedup window (seconds)
    COOLDOWN: Dict[str, float] = {
        "PERSON_ENTERED":  60,   # raised from 30 → fewer duplicate bursts
        "PERSON_RETURNED": 60,
        "PERSON_LEFT":     45,
        "INTRUDER":        30,
        "OBJ_ADDED":       90,
        "OBJ_REMOVED":     90,
        "SYSTEM":           0,
        "BASELINE":         0,
    }
    TG_MSG_HASH_DEDUP_SECS = 60   # same message content → drop within this window

    CAMERA_CONFIGS: List[dict] = [
        {"source": 0,                    "name": "Laptop Cam",  "enabled": True},
        {"source": "rtsp://admin:12345@192.168.1.50:554/Streaming/Channels/101",
         "name": "CCTV Cam 1", "enabled": True},
        {"source": "rtsp://admin:12345@192.168.1.50:554/Streaming/Channels/201",
         "name": "CCTV Cam 2", "enabled": True},
        {"source": "rtsp://admin:12345@192.168.1.50:554/Streaming/Channels/301",
         "name": "CCTV Cam 3", "enabled": True},
    ]

    # YOLO — per profile
    MODEL_NAME = {"LOW": "yolov8n.pt", "MEDIUM": "yolov8n.pt", "HIGH": "yolov8s.pt"}[HW_PROFILE]
    DEVICE     = "cuda" if CUDA_AVAILABLE else "cpu"

    FRAME_W, FRAME_H = 640, 360
    DET_W = {"LOW": 320, "MEDIUM": 416, "HIGH": 640}[HW_PROFILE]
    DET_H = {"LOW": 192, "MEDIUM": 256, "HIGH": 384}[HW_PROFILE]

    CONFIDENCE       = 0.50
    PROCESS_EVERY_N  = {"LOW": 5, "MEDIUM": 3, "HIGH": 2}[HW_PROFILE]
    TRACK_PERSIST    = True

    # Motion gate  [FIX-3]
    MOTION_THRESH_INIT = 300    # lower than v5 — fewer false-idles
    MOTION_CALIB_FRAMES = 30    # auto-calibrate threshold in first N frames
    IDLE_SKIP_EXTRA    = 2

    # Face recog
    FACE_MATCH_THRESH   = 0.60
    FACE_DETECT_MODEL   = "cnn" if CUDA_AVAILABLE else "hog"
    FACE_POOL_WORKERS   = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}[HW_PROFILE]
    FACE_RECHECK_CYCLES = {"LOW": 150, "MEDIUM": 100, "HIGH": 60}[HW_PROFILE]
    ABSENT_CYCLES_THRESH = 50
    KNOWN_FACES_DIR     = "faces/known"

    LOG_DIR       = Path("logs")
    FACES_DB_FILE = Path("logs/faces_db.json")
    ALARM_WAV     = "alarm.wav"

    # UI
    WINDOW_W   = 1600
    WINDOW_H   = 900
    TARGET_FPS = {"LOW": 24, "MEDIUM": 28, "HIGH": 30}[HW_PROFILE]
    SIDE_W     = 270
    EVENT_W    = 290
    TOP_H      = 58
    FOOTER_H   = 40

    DB_SAVE_SECS      = 45
    CAM_QUEUE_SIZE    = 1
    OVERLOAD_CPU_PCT  = 88.0
    GC_GEN0_FRAMES    = 25
    GC_GEN1_SECS      = 30

    # [FIX-5] UI animation
    SPARKLINE_SAMPLES = 40   # rolling FPS history per cam
    PULSE_HZ          = 1.2  # neon border pulse frequency


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveController:
    def __init__(self):
        self._lock       = threading.Lock()
        self.fps_target  = Config.TARGET_FPS
        self.det_w       = Config.DET_W
        self.det_h       = Config.DET_H
        self.skip_n      = Config.PROCESS_EVERY_N
        self.overloaded  = False
        self._hist: deque = deque(maxlen=8)
        self._last_t     = 0.0

    def update(self):
        now = time.time()
        if now - self._last_t < 2.0:
            return
        self._last_t = now
        cpu = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 50.0
        self._hist.append(cpu)
        avg = sum(self._hist) / len(self._hist)
        with self._lock:
            if avg > Config.OVERLOAD_CPU_PCT:
                self.fps_target = max(10, Config.TARGET_FPS // 3)
                self.skip_n     = Config.PROCESS_EVERY_N * 3
                self.det_w      = max(192, Config.DET_W // 2)
                self.det_h      = max(128, Config.DET_H // 2)
                self.overloaded = True
            elif avg > 72:
                self.fps_target = max(16, int(Config.TARGET_FPS * 0.7))
                self.skip_n     = Config.PROCESS_EVERY_N + 2
                self.det_w      = max(256, int(Config.DET_W * 0.75))
                self.det_h      = max(160, int(Config.DET_H * 0.75))
                self.overloaded = False
            else:
                self.fps_target = Config.TARGET_FPS
                self.skip_n     = Config.PROCESS_EVERY_N
                self.det_w      = Config.DET_W
                self.det_h      = Config.DET_H
                self.overloaded = False

    @property
    def frame_ms(self) -> int:
        return max(1, int(1000 / self.fps_target))


adaptive = AdaptiveController()


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING — async worker, fixed schema
# ══════════════════════════════════════════════════════════════════════════════

def _setup_logging():
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    app = logging.getLogger("surv.app")
    app.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    fh = logging.handlers.RotatingFileHandler(
        Config.LOG_DIR/"app.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    app.addHandler(sh); app.addHandler(fh)

    evt = logging.getLogger("surv.evt")
    evt.setLevel(logging.INFO)
    efh = logging.handlers.RotatingFileHandler(
        Config.LOG_DIR/"events.jsonl", maxBytes=10_000_000, backupCount=10, encoding="utf-8")
    efh.setFormatter(logging.Formatter("%(message)s"))
    evt.addHandler(efh)

    txt = logging.getLogger("surv.txt")
    txt.setLevel(logging.INFO)
    tfh = logging.handlers.RotatingFileHandler(
        Config.LOG_DIR/"events.log", maxBytes=10_000_000, backupCount=10, encoding="utf-8")
    tfh.setFormatter(logging.Formatter("%(message)s"))
    txt.addHandler(tfh)
    return app, evt, txt

app_log, evt_log, txt_log = _setup_logging()

_TXT_HDR_DONE = False
_TXT_COLS     = (22, 18, 18, 14, 22, 36)
_TXT_NAMES    = ("TIMESTAMP","EVENT","CAMERA","ID","NAME/OBJ","DETAIL")

def _tdiv(): return "+-" + "-+-".join("-"*w for w in _TXT_COLS) + "-+"
def _trow(c): return "| " + " | ".join(str(c[i])[:_TXT_COLS[i]].ljust(_TXT_COLS[i]) for i in range(6)) + " |"

_lq: queue.Queue = queue.Queue(maxsize=8192)
_event_ring: deque = deque(maxlen=80)
_event_ring_lock   = threading.Lock()

def _log_worker():
    global _TXT_HDR_DONE
    while True:
        try:
            item = _lq.get(timeout=5)
            if item is None: break
            ev, cam, person, obj, detail = item
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            evt_log.info(json.dumps({"ts":ts,"event":ev,"camera":cam,"person":person,"object":obj,"detail":detail}))
            if not _TXT_HDR_DONE:
                _TXT_HDR_DONE = True
                txt_log.info(f"\n{'='*120}\n{'AI SURVEILLANCE — EVENT LOG':^120}\n{'='*120}\n\n" +
                             _tdiv()+"\n"+_trow(_TXT_NAMES)+"\n"+_tdiv())
            txt_log.info(_trow((ts,ev,cam,person,obj,detail)))
            short = f"[{ts[11:]}] {ev:<14} {cam:<12} {person}"
            with _event_ring_lock:
                _event_ring.append((short, ev, time.time()))
            app_log.info(f"{ev:<18}| cam={cam} person={person} | {detail}")
        except queue.Empty: continue
        except Exception as e: print(f"[LOG] {e}")

_log_th = threading.Thread(target=_log_worker, daemon=True, name="LogWorker")
_log_th.start()

def log_event(ev, camera="", person="--", obj="--", detail=""):
    try: _lq.put_nowait((ev, camera, person, obj, detail))
    except queue.Full: pass

def export_csv(path="logs/events_export.csv"):
    src = Config.LOG_DIR/"events.jsonl"
    if not src.exists(): return
    with open(src, encoding="utf-8") as f, open(path,"w",newline="",encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["ts","event","camera","person","object","detail"])
        w.writeheader()
        for line in f:
            line = line.strip()
            if line:
                try: w.writerow(json.loads(line))
                except: pass
    app_log.info(f"CSV -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION QUEUE  [FIX-1] — triple-layer deduplication
# Layer 1: per-event-type cooldown per camera+person key
# Layer 2: SHA-256 hash of message text → drop if same hash seen within window
# Layer 3: in_scene flag in faces_db prevents re-firing on same visit
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Notification:
    kind:     str
    payload:  str
    caption:  str = ""
    priority: int = 5

class NotificationQueue:
    def __init__(self):
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=Config.TG_QUEUE_SIZE)
        self._cooldown: Dict[str, float] = {}   # event_key → next_allowed_time
        self._msg_hashes: Dict[str, float] = {} # sha256 → sent_time  [FIX-1 Layer 2]
        self._lock    = threading.Lock()
        self._counter = 0
        threading.Thread(target=self._worker, daemon=True, name="TG").start()
        app_log.info("NotificationQueue ready")

    def send_message(self, text: str, event_type: str = "SYSTEM",
                     camera: str = "", person: str = "", priority: int = 5):
        key      = f"{event_type}:{camera}:{person}"
        cooldown = Config.COOLDOWN.get(event_type, 45)
        msg_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        now      = time.time()

        with self._lock:
            # Layer 1: event-key cooldown
            if cooldown > 0 and self._cooldown.get(key, 0) > now:
                return
            # Layer 2: content hash dedup — exact same message within window
            if self._msg_hashes.get(msg_hash, 0) > now - Config.TG_MSG_HASH_DEDUP_SECS:
                return
            # Both checks passed — stamp both
            if cooldown > 0:
                self._cooldown[key] = now + cooldown
            self._msg_hashes[msg_hash] = now
            # Expire old hashes to prevent memory growth
            expired = [k for k, t in self._msg_hashes.items() if now - t > Config.TG_MSG_HASH_DEDUP_SECS * 2]
            for k in expired: del self._msg_hashes[k]

        self._enqueue(Notification("message", text, priority=priority))

    def send_photo(self, path: str, caption: str = "", priority: int = 6):
        self._enqueue(Notification("photo", path, caption=caption, priority=priority))

    def send_alert(self, text: str, photo_path: Optional[str] = None,
                   event_type: str = "SYSTEM", camera: str = "",
                   person: str = "", priority: int = 5):
        self.send_message(text, event_type=event_type, camera=camera,
                          person=person, priority=priority)
        if photo_path and os.path.exists(photo_path):
            self.send_photo(photo_path, caption=_caption(text), priority=priority+1)

    def _enqueue(self, n: Notification):
        try:
            with self._lock:
                self._counter += 1
                cnt = self._counter
            self._q.put_nowait((n.priority, cnt, n))
        except queue.Full:
            app_log.warning("TG queue full")

    def _worker(self):
        while True:
            try:
                _, _, n = self._q.get(timeout=5)
                self._dispatch(n)
                self._q.task_done()
            except queue.Empty: continue
            except Exception as e: app_log.error(f"TG worker: {e}")

    def _dispatch(self, n: Notification):
        for attempt in range(1, Config.TG_MAX_RETRIES+1):
            try:
                if n.kind == "message":
                    requests.post(
                        f"https://api.telegram.org/bot{Config.BOT_TOKEN}/sendMessage",
                        json={"chat_id":Config.CHAT_ID,"text":n.payload,"parse_mode":"HTML"},
                        timeout=Config.TG_TIMEOUT).raise_for_status()
                else:
                    if not os.path.exists(n.payload): return
                    with open(n.payload,"rb") as f:
                        requests.post(
                            f"https://api.telegram.org/bot{Config.BOT_TOKEN}/sendPhoto",
                            files={"photo":f},
                            data={"chat_id":Config.CHAT_ID,"caption":n.caption},
                            timeout=Config.TG_TIMEOUT).raise_for_status()
                return
            except Exception as e:
                app_log.warning(f"[TG] attempt {attempt}: {e}")
                if attempt < Config.TG_MAX_RETRIES:
                    time.sleep(Config.TG_RETRY_DELAY * attempt)
        app_log.error("[TG] all retries failed")

def _caption(t):
    for l in t.splitlines():
        if l.strip(): return l.strip()[:200]
    return ""

# Telegram message templates
def _tg_person_enter(cam,pid,name,visits,conf,ts):
    icon = "\U0001F6A8" if "Intruder" in name else "\U0001F7E2"
    lbl  = "INTRUDER ALERT" if "Intruder" in name else "PERSON ENTERED"
    return (f"{icon} <b>{lbl}</b>\n\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            f"\U0001F4F7 <b>Camera</b>: {cam}\n\U0001F194 <b>ID</b>: {pid}\n"
            f"\U0001F464 <b>Name</b>: {name}\n\U0001F501 <b>Visit #</b>: {visits}\n"
            f"\U0001F3AF <b>Conf</b>: {conf:.0%}\n\u23F0 <b>Time</b>: {ts}\n"
            f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")

def _tg_person_left(cam,pid,name,ts):
    return (f"\U0001F534 <b>PERSON LEFT</b>\n\u2500"*1 +
            f"\U0001F4F7 Camera: {cam}\n\U0001F194 ID: {pid}\n"
            f"\U0001F464 Name: {name}\n\u23F0 Time: {ts}")

def _tg_obj(cam,ev,label,before,after,ts):
    icon = "\U0001F4E6" if ev=="OBJ_ADDED" else "\u26A0\uFE0F"
    verb = "ADDED" if ev=="OBJ_ADDED" else "REMOVED"
    return (f"{icon} <b>OBJECT {verb}</b>\n"
            f"\U0001F4F7 Camera: {cam}\n\U0001F50D Object: {label}\n"
            f"\U0001F4CA {before} \u2192 {after}\n\u23F0 {ts}")

notif_queue = NotificationQueue()


# ══════════════════════════════════════════════════════════════════════════════
# FACES DATABASE
# ══════════════════════════════════════════════════════════════════════════════

for d in ["logs","faces/known","faces/unknown","faces/captured"]:
    Path(d).mkdir(parents=True, exist_ok=True)

_fdb_lock  = threading.RLock()
_next_pid  = 1

def _load_db():
    if not Config.FACES_DB_FILE.exists(): return {}
    try:
        with open(Config.FACES_DB_FILE, encoding="utf-8") as f:
            db = json.load(f)
        if FACE_RECOG_AVAILABLE:
            for d in db.values():
                if "encoding" in d: d["encoding"] = np.array(d["encoding"])
        return db
    except Exception as e:
        app_log.error(f"DB load: {e}"); return {}

def _save_db(db):
    tmp = Config.FACES_DB_FILE.with_suffix(".tmp")
    out = {}
    for pid, d in db.items():
        out[pid] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in d.items()}
    try:
        with open(tmp,"w",encoding="utf-8") as f: json.dump(out,f,indent=2)
        if IS_WINDOWS and Config.FACES_DB_FILE.exists(): Config.FACES_DB_FILE.unlink()
        tmp.rename(Config.FACES_DB_FILE)
    except Exception as e: app_log.error(f"DB save: {e}")

faces_db = _load_db()
for _p in faces_db:
    try:
        n = int(_p[1:])
        if n >= _next_pid: _next_pid = n+1
    except: pass

def _new_pid():
    global _next_pid
    p = f"P{_next_pid}"; _next_pid += 1; return p

def preload_known():
    if not FACE_RECOG_AVAILABLE: return
    by_name = {d["name"]:pid for pid,d in faces_db.items() if d.get("known")}
    loaded  = []
    p = Path(Config.KNOWN_FACES_DIR)
    if not p.exists(): return
    for fp in p.iterdir():
        if fp.suffix.lower() not in (".jpg",".jpeg",".png"): continue
        name = fp.stem
        try:
            img  = face_recognition.load_image_file(str(fp))
            encs = face_recognition.face_encodings(img)
            if not encs: continue
            now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name in by_name:
                faces_db[by_name[name]]["encoding"] = encs[0]
            else:
                pid = _new_pid()
                faces_db[pid] = {"name":name,"encoding":encs[0],"first_seen":now,
                                 "last_seen":now,"visit_count":0,"known":True,
                                 "photo":str(fp),"in_scene":False}
            loaded.append(name)
        except Exception as e: app_log.error(f"Load face {fp.name}: {e}")
    _save_db(faces_db)
    app_log.info(f"Known faces: {loaded or 'none'}")

preload_known()

# Encoding cache (rebuilt only on change)
_enc_arr:   Optional[np.ndarray] = None
_enc_pids:  List[str]            = []
_enc_dirty  = True

def _rebuild_enc_cache():
    global _enc_arr, _enc_pids, _enc_dirty
    with _fdb_lock:
        pids = [p for p,d in faces_db.items() if "encoding" in d]
        _enc_pids = pids
        _enc_arr  = np.array([faces_db[p]["encoding"] for p in pids]) if pids else None
        _enc_dirty = False

def match_face(enc):
    global _enc_dirty
    if _enc_dirty: _rebuild_enc_cache()
    if _enc_arr is None: return _register_face(enc)
    dists = face_recognition.face_distance(_enc_arr, enc)
    bi    = int(np.argmin(dists))
    if dists[bi] <= Config.FACE_MATCH_THRESH:
        pid = _enc_pids[bi]
        with _fdb_lock: name = faces_db[pid]["name"]
        return pid, name, False
    return _register_face(enc)

def _register_face(enc):
    global _enc_dirty
    with _fdb_lock:
        pid  = _new_pid()
        name = f"Intruder-{pid}"
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        faces_db[pid] = {"name":name,"encoding":enc,"first_seen":now,"last_seen":now,
                         "visit_count":0,"known":False,"photo":None,"in_scene":False}
        _enc_dirty = True
        _save_db(faces_db)
    return pid, name, True

def async_face(rgb):
    if not FACE_RECOG_AVAILABLE: return None, None
    try:
        locs = face_recognition.face_locations(rgb, model=Config.FACE_DETECT_MODEL)
        if not locs: return None, None
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs: return None, None
        pid, name, _ = match_face(encs[0])
        return pid, name
    except Exception as e: app_log.debug(f"face: {e}"); return None, None

face_pool = ThreadPoolExecutor(max_workers=Config.FACE_POOL_WORKERS, thread_name_prefix="Face")


# ══════════════════════════════════════════════════════════════════════════════
# MOTION DETECTOR  [FIX-3] — auto-calibrating threshold
# ══════════════════════════════════════════════════════════════════════════════

class MotionDetector:
    def __init__(self):
        self._mog   = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=32, detectShadows=False)
        self._sw    = 160; self._sh = 90
        self._thresh = Config.MOTION_THRESH_INIT
        self._calib_scores: deque = deque(maxlen=Config.MOTION_CALIB_FRAMES)
        self._calibrated = False

    def has_motion(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (self._sw, self._sh), interpolation=cv2.INTER_NEAREST)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask  = self._mog.apply(gray)
        score = int(np.sum(mask > 120))

        # Auto-calibrate: set threshold = 3× median of first N idle scores
        if not self._calibrated:
            self._calib_scores.append(score)
            if len(self._calib_scores) >= Config.MOTION_CALIB_FRAMES:
                baseline        = sorted(self._calib_scores)[len(self._calib_scores)//2]
                self._thresh    = max(200, baseline * 3)
                self._calibrated = True
                app_log.info(f"[Motion] threshold calibrated → {self._thresh}")

        return score > self._thresh


# ══════════════════════════════════════════════════════════════════════════════
# YOLO — per-camera instance  [FIX-2] — no shared global lock
# ══════════════════════════════════════════════════════════════════════════════

_yolo_load_lock = threading.Lock()   # only used during initial load

def make_camera_yolo() -> Optional[object]:
    """Each camera thread gets its own YOLO reference.
    On CPU, separate instances share no state → no cross-thread blocking.
    On CUDA, all instances share GPU — protected by torch's own stream queue."""
    if not YOLO_AVAILABLE: return None
    with _yolo_load_lock:    # serialise model download/load, not inference
        app_log.info(f"[YOLO] Loading {Config.MODEL_NAME} on {Config.DEVICE}")
        m = YOLO(Config.MODEL_NAME)
        dummy = np.zeros((Config.DET_H, Config.DET_W, 3), dtype=np.uint8)
        try: m.predict(source=dummy, device=Config.DEVICE, verbose=False, half=CUDA_AVAILABLE)
        except Exception as e: app_log.warning(f"YOLO warm-up: {e}")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PersonInfo:
    pid:            str
    name:           str
    present:        bool         = False
    absent_cycles:  int          = 0
    last_box:       Optional[Tuple] = None
    face_check_cd:  int          = 0

class CameraState:
    def __init__(self, cam_id: int, cfg: dict):
        self.cam_id  = cam_id
        self.source  = cfg["source"]
        self.name    = cfg["name"]
        self.enabled = cfg.get("enabled", True)
        self.cap: Optional[cv2.VideoCapture] = None
        self.online    = False
        self.frame_cnt = 0
        self.fps_inst  = 0.0
        self._fps_t    = time.perf_counter()
        self._fps_cnt  = 0
        self.frame_q: queue.Queue = queue.Queue(maxsize=Config.CAM_QUEUE_SIZE)
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_dets:  List[dict] = []
        self.frame_lock = threading.Lock()
        self.baseline_saved  = False
        self.startup_time    = time.time()
        self.baseline_counts: Counter = Counter()
        self.before_img = str(Config.LOG_DIR / f"before_cam{cam_id}.jpg")
        self.after_img  = str(Config.LOG_DIR / f"after_cam{cam_id}.jpg")
        self.track_to_pid:    Dict[int, str]        = {}
        self.pid_info:        Dict[str, PersonInfo] = {}
        self.present_pids:    set                   = set()
        self.pending_futures: Dict[int, object]     = {}
        self.obj_missing_since: Dict[str, float] = {}
        self.obj_added_since:   Dict[str, float] = {}
        self.motion       = MotionDetector()
        self.last_motion  = time.time()
        self.idle_cnt     = 0
        self.warning_msg  = ""
        self.warning_time = 0.0
        self.display_tile: Optional[np.ndarray] = None
        self.tile_dirty   = True
        self.persons_detected = 0
        self.persons_left     = 0
        self.uptime_start     = time.time()
        # [FIX-5] FPS sparkline
        self.fps_spark: deque = deque([0.0]*Config.SPARKLINE_SAMPLES,
                                      maxlen=Config.SPARKLINE_SAMPLES)
        self._spark_t   = time.time()
        # [FIX-5] Detection flash animation
        self.det_flash_t  = 0.0   # set when person detected
        self.det_flash_pid = ""

    def connect(self) -> bool:
        try:
            app_log.info(f"[{self.name}] connect {self.source!r}")
            if isinstance(self.source, int):
                backend = cv2.CAP_V4L2 if IS_LINUX else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.source, backend)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_W)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_H)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                                      "rtsp_transport;tcp|buffer_size;1048576")
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.online = self.cap.isOpened()
            if self.online:
                ret, frame = self.cap.read()
                if not ret or frame is None: self.online = False
            app_log.info(f"[{self.name}] online={self.online}")
            return self.online
        except Exception as e:
            app_log.error(f"[{self.name}] connect: {e}")
            self.online = False; return False

    def release(self):
        if self.cap: self.cap.release(); self.cap = None

    @property
    def uptime_str(self):
        s = int(time.time()-self.uptime_start)
        h,r = divmod(s,3600); m,s = divmod(r,60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# EVENT HANDLERS  [FIX-1] — in_scene guard prevents duplicate arrival events
# ══════════════════════════════════════════════════════════════════════════════

def on_person_arrived(cs: CameraState, pid: str, name: str,
                       frame: np.ndarray, box: Tuple, conf: float):
    # [FIX-1] LAYER 3: check global in_scene flag FIRST
    with _fdb_lock:
        db = faces_db.get(pid)
        if db is None: return
        if db.get("in_scene", False):
            return   # already in scene — do NOT fire again
        db["in_scene"] = True
        db["last_seen"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db["visit_count"] = db.get("visit_count", 0) + 1
        visits = db["visit_count"]

    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = "PERSON_ENTERED" if visits == 1 else "PERSON_RETURNED"
    log_event(event, camera=cs.name, person=name, obj="person",
              detail=f"visits={visits} conf={conf:.2f}")
    cs.persons_detected += 1
    cs.det_flash_t   = time.time()
    cs.det_flash_pid = pid

    x1,y1,x2,y2 = box
    crop = frame[y1:y2, x1:x2]
    photo_path = None
    if crop.size > 0:
        photo_path = f"faces/captured/{pid}_{int(time.time())}.jpg"
        cv2.imwrite(photo_path, crop)
        with _fdb_lock:
            if not db.get("known") and not db.get("photo"):
                db["photo"] = photo_path

    _save_db(faces_db)
    _alarm()

    msg = _tg_person_enter(cs.name, pid, name, visits, conf, now)
    notif_queue.send_alert(msg, photo_path=photo_path,
                           event_type="INTRUDER" if "Intruder" in name else "PERSON_ENTERED",
                           camera=cs.name, person=pid)
    cs.warning_msg  = f"{'[INTRUDER]' if 'Intruder' in name else '[ARRIVED]'}: {name}"
    cs.warning_time = time.time()
    cs.tile_dirty   = True

def on_person_left(cs: CameraState, pid: str, name: str, frame: np.ndarray):
    # [FIX-1] Clear in_scene so next arrival fires correctly
    with _fdb_lock:
        db = faces_db.get(pid)
        if db: db["in_scene"] = False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.imwrite(cs.after_img, frame)
    log_event("PERSON_LEFT", camera=cs.name, person=name, detail=f"last_seen={now}")
    cs.persons_left += 1
    _alarm()
    notif_queue.send_message(_tg_person_left(cs.name,pid,name,now),
                             event_type="PERSON_LEFT", camera=cs.name, person=pid)
    cs.warning_msg  = f"[LEFT]: {name}"
    cs.warning_time = time.time()
    cs.tile_dirty   = True

def on_obj_event(cs, event, label, before, after, frame):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.imwrite(cs.after_img, frame)
    log_event(event, camera=cs.name, obj=label, detail=f"{before}->{after}")
    _alarm()
    notif_queue.send_message(_tg_obj(cs.name,event,label,before,after,now),
                             event_type=event, camera=cs.name)
    verb = "ADDED" if event=="OBJ_ADDED" else "REMOVED"
    cs.warning_msg  = f"[{verb}] {label.upper()}"
    cs.warning_time = time.time()
    cs.tile_dirty   = True

def _alarm():
    if IS_WINDOWS and WINSOUND_AVAILABLE and os.path.exists(Config.ALARM_WAV):
        threading.Thread(
            target=lambda: winsound.PlaySound(Config.ALARM_WAV, winsound.SND_ASYNC),
            daemon=True).start()
    elif IS_LINUX and os.path.exists(Config.ALARM_WAV):
        player = shutil.which("paplay") or shutil.which("aplay")
        if player:
            threading.Thread(
                target=lambda: subprocess.run([player, Config.ALARM_WAV],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA THREAD  [FIX-2] — no global YOLO lock, UI fully decoupled
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_pid(cs, pid, name):
    if pid not in cs.pid_info: cs.pid_info[pid] = PersonInfo(pid=pid, name=name)
    else: cs.pid_info[pid].name = name
    return cs.pid_info[pid]

def camera_thread(cs: CameraState):
    yolo    = make_camera_yolo()   # [FIX-2] own instance, no shared lock
    gc_ctr  = 0
    backoff = 3.0

    while not cs.connect():
        app_log.warning(f"[{cs.name}] retry in {backoff:.0f}s")
        time.sleep(backoff); backoff = min(backoff*1.5, 30.0)

    while True:
        if not cs.online:
            time.sleep(3); cs.release(); backoff = 3.0
            while not cs.connect():
                time.sleep(backoff); backoff = min(backoff*1.5, 30.0)
            log_event("CAM_RECONNECT", camera=cs.name); continue

        ret, frame = cs.cap.read()
        if not ret or frame is None: cs.online = False; continue

        frame = cv2.flip(frame, 1)
        cs.frame_cnt += 1

        cs._fps_cnt += 1
        t = time.perf_counter(); e = t - cs._fps_t
        if e >= 1.0:
            cs.fps_inst = cs._fps_cnt/e; cs._fps_cnt=0; cs._fps_t=t
            # [FIX-5] Update sparkline once per second
            cs.fps_spark.append(cs.fps_inst)

        if not cs.baseline_saved and time.time()-cs.startup_time > 5:
            cv2.imwrite(cs.before_img, frame)
            cs.baseline_saved = True
            log_event("BASELINE", camera=cs.name, detail="saved")
            notif_queue.send_message(
                f"\u2705 <b>CAMERA ONLINE</b>\n\U0001F4F7 {cs.name}\n"
                f"\U0001F4BB Profile: {HW_PROFILE}\n\u23F0 Baseline captured",
                event_type="BASELINE", camera=cs.name)

        # Motion gate
        has_motion = cs.motion.has_motion(frame)
        if has_motion: cs.last_motion=time.time(); cs.idle_cnt=0
        else: cs.idle_cnt+=1

        idle = (time.time()-cs.last_motion) > 3.0 and not has_motion
        skip_n = adaptive.skip_n + (Config.IDLE_SKIP_EXTRA if idle else 0)

        if cs.frame_cnt % skip_n != 0:
            with cs.frame_lock: cs.latest_frame=frame; cs.tile_dirty=has_motion
            continue

        dw, dh = adaptive.det_w, adaptive.det_h
        small  = cv2.resize(frame, (dw,dh), interpolation=cv2.INTER_LINEAR)
        new_dets=[]; cur_objs=[]; tid_seen=set()

        if yolo:
            try:
                # [FIX-2] No global lock — each camera calls its own instance
                results = yolo.track(source=small, conf=Config.CONFIDENCE,
                                     device=Config.DEVICE, persist=Config.TRACK_PERSIST,
                                     verbose=False, half=CUDA_AVAILABLE,
                                     imgsz=max(dw,dh))
                boxes = results[0].boxes
                sx = Config.FRAME_W/dw; sy = Config.FRAME_H/dh

                for box in boxes:
                    cv = float(box.conf[0])
                    if cv < Config.CONFIDENCE: continue
                    cls   = int(box.cls[0]); label = yolo.names[cls]
                    bx1,by1,bx2,by2 = box.xyxy[0]
                    x1=int(max(0,bx1*sx)); y1=int(max(0,by1*sy))
                    x2=int(min(Config.FRAME_W-1,bx2*sx)); y2=int(min(Config.FRAME_H-1,by2*sy))
                    cur_objs.append(label)
                    pid, disp = None, label

                    if label == "person":
                        tid = None
                        if box.id is not None:
                            try: tid = int(box.id[0])
                            except: pass
                        if tid is not None:
                            tid_seen.add(tid)
                            fut = cs.pending_futures.get(tid)
                            if fut and fut.done():
                                np_pid, np_name = fut.result()
                                del cs.pending_futures[tid]
                                if np_pid:
                                    old = cs.track_to_pid.get(tid)
                                    cs.track_to_pid[tid] = np_pid
                                    _ensure_pid(cs, np_pid, np_name)
                                    if old and old!=np_pid: cs.present_pids.discard(old)
                            if tid in cs.track_to_pid:
                                pid = cs.track_to_pid[tid]
                                if pid not in faces_db: del cs.track_to_pid[tid]; pid=None
                                else:
                                    info = cs.pid_info.get(pid)
                                    if info:
                                        disp = f"{pid} {info.name}"
                                        info.face_check_cd -= 1
                                        if info.face_check_cd<=0 and tid not in cs.pending_futures:
                                            info.face_check_cd = Config.FACE_RECHECK_CYCLES
                                            crop = frame[y1:y2,x1:x2]
                                            if crop.size>0:
                                                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                                cs.pending_futures[tid] = face_pool.submit(async_face,rgb)
                            else:
                                if tid not in cs.pending_futures:
                                    crop=frame[y1:y2,x1:x2]
                                    if crop.size>0:
                                        rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
                                        cs.pending_futures[tid]=face_pool.submit(async_face,rgb)
                                disp = "Identifying..."
                        if pid:
                            info = cs.pid_info[pid]
                            info.last_box=(x1,y1,x2,y2); info.absent_cycles=0
                            if not info.present:
                                info.present=True; cs.present_pids.add(pid)
                                on_person_arrived(cs,pid,info.name,frame,(x1,y1,x2,y2),cv)
                    new_dets.append({"label":label,"conf":cv,"box":(x1,y1,x2,y2),"disp":disp,"pid":pid})
            except Exception as e: app_log.error(f"[{cs.name}] YOLO: {e}")

        # Departure
        active = {cs.track_to_pid[t] for t in tid_seen if t in cs.track_to_pid}
        for pid in list(cs.present_pids):
            if pid not in active:
                info = cs.pid_info[pid]
                info.absent_cycles+=1
                if info.absent_cycles>=Config.ABSENT_CYCLES_THRESH:
                    cs.present_pids.discard(pid); info.present=False; info.absent_cycles=0
                    for t in [t for t,p in cs.track_to_pid.items() if p==pid]:
                        del cs.track_to_pid[t]
                    on_person_left(cs,pid,info.name,frame)

        # Object change
        if cs.baseline_saved:
            now_ts=time.time(); cur=Counter(cur_objs)
            for obj,cnt in list(cs.baseline_counts.items()):
                cc=cur.get(obj,0)
                if cc<cnt:
                    cs.obj_missing_since.setdefault(obj,now_ts)
                    if now_ts-cs.obj_missing_since[obj]>=1.5:
                        on_obj_event(cs,"OBJ_REMOVED",obj,cnt,cc,frame)
                        cs.baseline_counts[obj]=cc
                        if cc==0: del cs.baseline_counts[obj]
                        cs.obj_missing_since.pop(obj,None)
                else: cs.obj_missing_since.pop(obj,None)
            for obj,cc in cur.items():
                cnt=cs.baseline_counts.get(obj,0)
                if cc>cnt:
                    cs.obj_added_since.setdefault(obj,now_ts)
                    if now_ts-cs.obj_added_since[obj]>=1.5:
                        on_obj_event(cs,"OBJ_ADDED",obj,cnt,cc,frame)
                        cs.baseline_counts[obj]=cc
                        cs.obj_added_since.pop(obj,None)
                else: cs.obj_added_since.pop(obj,None)

        with cs.frame_lock:
            cs.latest_dets=new_dets; cs.latest_frame=frame; cs.tile_dirty=True

        gc_ctr+=1
        if gc_ctr>=Config.GC_GEN0_FRAMES: gc_ctr=0; gc.collect(0)


# ══════════════════════════════════════════════════════════════════════════════
# ██╗   ██╗██╗    ██████╗ ██████╗  █████╗ ██╗    ██╗███████╗██████╗ ███████╗
# ██║   ██║██║    ██╔══██╗██╔══██╗██╔══██╗██║    ██║██╔════╝██╔══██╗██╔════╝
# ██║   ██║██║    ██║  ██║██████╔╝███████║██║ █╗ ██║█████╗  ██████╔╝███████╗
# ██║   ██║██║    ██║  ██║██╔══██╗██╔══██║██║███╗██║██╔══╝  ██╔══██╗╚════██║
# ╚██████╔╝██║    ██████╔╝██║  ██║██║  ██║╚███╔███╔╝███████╗██║  ██║███████║
#  ╚═════╝ ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚══════╝
# [FIX-5]  Dynamic cyberpunk UI system
# ══════════════════════════════════════════════════════════════════════════════

# ── Palette (BGR) ────────────────────────────────────────────────────────────
C_BG      = (8,    8,   15)
C_PANEL   = (12,  14,   24)
C_ACCENT  = (0,  230,  180)   # neon teal
C_BLUE    = (0,  170,  255)   # electric blue
C_PURPLE  = (200, 80,  255)   # neon purple
C_GREEN   = (0,  220,   90)
C_RED     = (25,  35,  215)
C_ORANGE  = (0,  160,  255)
C_WHITE   = (230,235,  245)
C_GRAY    = (70,  78,   92)
C_DIM     = (28,  32,   46)
C_AMBER   = (0,  200,  255)
C_WARN    = (20,  20,  180)
C_OK_BG   = (0,   60,   20)
C_BLACK   = (4,    4,   10)
C_DARK    = (6,    6,   12)

_PID_PALETTE = [
    (0,255,180),(255,200,0),(200,0,255),(0,200,255),
    (255,80,0),(80,255,0),(0,80,255),(255,0,150),
    (0,255,80),(150,0,255),(255,150,0),(0,150,255),
]
_pid_colors: Dict[str,tuple] = {}
def pid_col(pid): 
    if pid not in _pid_colors: _pid_colors[pid]=_PID_PALETTE[len(_pid_colors)%len(_PID_PALETTE)]
    return _pid_colors[pid]

# ── Drawing primitives ────────────────────────────────────────────────────────

def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.72):
    """Fast in-place alpha-blend rectangle."""
    x1=max(0,x1); y1=max(0,y1)
    x2=min(img.shape[1],x2); y2=min(img.shape[0],y2)
    if x2<=x1 or y2<=y1: return
    roi = img[y1:y2, x1:x2]
    ov  = np.empty_like(roi); ov[:] = color
    cv2.addWeighted(ov, alpha, roi, 1.0-alpha, 0, roi)
    img[y1:y2, x1:x2] = roi

def text(img, s, x, y, scale, color, thick=1, aa=True):
    cv2.putText(img, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA if aa else cv2.LINE_4)

def glow_text(img, s, x, y, scale, color, thick=1):
    """Text with soft neon glow (shadow behind)."""
    dr,dg,db = max(0,color[0]-80), max(0,color[1]-80), max(0,color[2]-80)
    cv2.putText(img, s, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (dr,dg,db), thick+2, cv2.LINE_AA)
    cv2.putText(img, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def corner_box(img, x1, y1, x2, y2, color, sz=14, th=2):
    """Animated corner brackets — sz controls arm length."""
    for pts in [
        ((x1,y1+sz),(x1,y1),(x1+sz,y1)),
        ((x2-sz,y1),(x2,y1),(x2,y1+sz)),
        ((x1,y2-sz),(x1,y2),(x1+sz,y2)),
        ((x2-sz,y2),(x2,y2),(x2,y2-sz)),
    ]:
        cv2.polylines(img,[np.array(pts,dtype=np.int32)],False,color,th,cv2.LINE_AA)

def neon_rect(img, x1, y1, x2, y2, color, pulse: float = 1.0, th=1):
    """Rectangle that pulses in brightness (pulse 0..1)."""
    c = tuple(int(v*pulse) for v in color)
    cv2.rectangle(img, (x1,y1), (x2,y2), c, th)

def draw_label_bg(img, txt, x, y, color, scale=0.40):
    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad=4
    bx1=x; by1=max(0,y-th-pad*2); bx2=x+tw+pad*2; by2=y
    alpha_rect(img, bx1, by1, bx2, by2, color, 0.88)
    cv2.putText(img, txt, (bx1+pad, by2-pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, C_BLACK, 1, cv2.LINE_AA)

def scanline_sweep(img, t: float, speed=0.3, width=4, alpha=0.12):
    """Animated bright scanline sweeping top-to-bottom."""
    h = img.shape[0]
    y = int((t * speed * h) % h)
    for dy in range(width):
        yy = (y+dy) % h
        a  = alpha * (1.0 - dy/width)
        row = img[yy:yy+1]
        bright = np.clip(row.astype(np.float32) * (1.0+a*2), 0, 255).astype(np.uint8)
        img[yy:yy+1] = bright

def draw_sparkline(img, x, y, w, h, values, color, bg=C_DIM):
    """Mini FPS sparkline bar chart."""
    alpha_rect(img, x, y, x+w, y+h, bg, 0.65)
    if not any(v>0 for v in values): return
    maxv = max(max(values), 1.0)
    bw   = max(1, w // len(values))
    for i, v in enumerate(values):
        bh  = int((v/maxv) * h)
        bx1 = x + i*bw
        bx2 = bx1+bw-1
        by2 = y+h; by1 = max(y, by2-bh)
        # color ramp: red < 30%, orange < 60%, green above
        ratio = v/maxv
        bc = C_RED if ratio<0.3 else (C_ORANGE if ratio<0.6 else color)
        cv2.rectangle(img, (bx1,by1),(bx2,by2), bc, -1)


# ══════════════════════════════════════════════════════════════════════════════
# TILE RENDERER — [FIX-5] dynamic animated cyberpunk tile
# ══════════════════════════════════════════════════════════════════════════════

def draw_tile(cs: CameraState, tw: int, th: int, anim_t: float) -> np.ndarray:
    with cs.frame_lock:
        frame = cs.latest_frame
        dets  = list(cs.latest_dets)
        dirty = cs.tile_dirty
        if dirty: cs.tile_dirty = False

    # Offline plate
    if frame is None:
        if cs.display_tile is None or cs.display_tile.shape[:2] != (th,tw):
            tile = np.full((th,tw,3), C_BG, dtype=np.uint8)
            cx,cy = tw//2, th//2
            alpha_rect(tile, cx-200,cy-55,cx+200,cy+55, C_PANEL, 0.92)
            # Pulsing border on offline camera
            pulse = 0.4 + 0.3*math.sin(anim_t*2)
            neon_rect(tile, cx-200,cy-55,cx+200,cy+55, C_RED, pulse, 1)
            glow_text(tile, cs.name, cx-len(cs.name)*9, cy-12, 0.65, C_GRAY)
            glow_text(tile, "[[ OFFLINE  //  CONNECTING ]]", cx-168, cy+24, 0.44, C_RED)
            cs.display_tile = tile
        return cs.display_tile

    if (not dirty and cs.display_tile is not None
            and cs.display_tile.shape[:2]==(th,tw)):
        return cs.display_tile

    tile = cv2.resize(frame, (tw,th), interpolation=cv2.INTER_LINEAR)
    sx = tw/Config.FRAME_W; sy = th/Config.FRAME_H

    # [FIX-5] Animated scanline sweep
    scanline_sweep(tile, anim_t, speed=0.25, width=3, alpha=0.09)

    # [FIX-5] Pulsing neon border on camera with persons present
    if cs.present_pids:
        pulse = 0.7 + 0.3*math.sin(anim_t * Config.PULSE_HZ * math.pi * 2)
        col   = tuple(int(v*pulse) for v in C_ACCENT)
        cv2.rectangle(tile, (1,1), (tw-2,th-2), col, 2)
        # second inner line for glow depth
        col2 = tuple(int(v*pulse*0.4) for v in C_ACCENT)
        cv2.rectangle(tile, (3,3), (tw-4,th-4), col2, 1)

    # Detection flash: white edge flash for 0.4s after new person
    flash_age = time.time()-cs.det_flash_t
    if flash_age < 0.4:
        fstr = 1.0 - flash_age/0.4
        fc   = tuple(int(255*fstr) for _ in range(3))
        cv2.rectangle(tile, (0,0),(tw-1,th-1), fc, 3)

    # Detections
    for d in dets:
        ox1,oy1,ox2,oy2 = d["box"]
        tx1=int(ox1*sx); ty1=int(oy1*sy)
        tx2=int(ox2*sx); ty2=int(oy2*sy)
        pid   = d.get("pid")
        color = pid_col(pid) if pid else C_ACCENT

        if pid:
            # [FIX-5] Animated corner brackets — arm length pulses slightly
            age  = time.time()-cs.det_flash_t
            sz   = 14 + int(6*max(0,1.0-age/0.6)) if pid==cs.det_flash_pid else 14
            corner_box(tile, tx1,ty1,tx2,ty2, color, sz=sz, th=2)
        else:
            cv2.rectangle(tile,(tx1,ty1),(tx2,ty2),color,1)

        draw_label_bg(tile, f"{d['disp']}  {d['conf']:.0%}", tx1, ty1, color, 0.38)

    # ── Top bar ───────────────────────────────────────────────────────────────
    bar_h = 34
    alpha_rect(tile, 0,0,tw,bar_h, C_BLACK, 0.85)
    cv2.line(tile,(0,bar_h),(tw,bar_h),C_DIM,1)

    # Status dot (blinking if persons present)
    dot_c = C_GREEN if cs.online else C_RED
    if cs.online and cs.present_pids:
        dot_c = C_ACCENT if int(anim_t*3)%2==0 else C_GREEN
    cv2.circle(tile,(14,bar_h//2),5,dot_c,-1,cv2.LINE_AA)
    cv2.circle(tile,(14,bar_h//2),5,tuple(v//2 for v in dot_c),1,cv2.LINE_AA)

    glow_text(tile, cs.name, 26, bar_h-8, 0.46, C_WHITE)

    # Live badge
    if cs.online:
        lbw = 36; lbh = 18; lbx = tw-lbw-42; lby = 4
        alpha_rect(tile, lbx,lby,lbx+lbw,lby+lbh, C_RED, 0.85)
        blink = int(anim_t*2)%2
        lc = (50,50,255) if blink else C_WHITE
        cv2.putText(tile,"LIVE",(lbx+5,lby+13),cv2.FONT_HERSHEY_SIMPLEX,0.34,lc,1,cv2.LINE_AA)

    # CAM badge
    badge = f"CAM{cs.cam_id+1}"
    bw2   = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)[0][0]
    alpha_rect(tile, tw-bw2-14,0,tw,20, C_BLUE, 0.80)
    cv2.putText(tile,badge,(tw-bw2-7,15),cv2.FONT_HERSHEY_SIMPLEX,0.36,C_BLACK,1,cv2.LINE_AA)

    # Right stats
    rstats = f"FPS:{cs.fps_inst:4.1f}  OBJ:{len(dets)}  PRS:{len(cs.present_pids)}"
    rsz    = cv2.getTextSize(rstats,cv2.FONT_HERSHEY_SIMPLEX,0.35,1)[0][0]
    cv2.putText(tile,rstats,(tw-rsz-52,bar_h-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,C_ACCENT,1,cv2.LINE_AA)

    # Scanning indicator
    if not cs.baseline_saved:
        dots  = "." * (int(anim_t*4)%4+1)
        stext = f">> SCANNING{dots}"
        alpha_rect(tile, tw//2-80,bar_h+2,tw//2+80,bar_h+26, C_AMBER, 0.78)
        cv2.putText(tile,stext,(tw//2-68,bar_h+20),cv2.FONT_HERSHEY_SIMPLEX,0.44,C_BLACK,1,cv2.LINE_AA)

    # Idle indicator
    if (time.time()-cs.last_motion)>3.0:
        cv2.putText(tile,"[ZZZ IDLE]",(tw//2-38,bar_h+22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,C_GRAY,1,cv2.LINE_AA)

    # [FIX-5] Mini sparkline bottom-left
    spark_x = 4; spark_y = th-24; spark_w = 80; spark_h = 18
    draw_sparkline(tile, spark_x, spark_y, spark_w, spark_h,
                   list(cs.fps_spark), C_GREEN)

    # Warning banner
    age = time.time()-cs.warning_time
    if age < 5:
        fade = max(0.0, 1.0-(age/5.0))
        is_danger = any(x in cs.warning_msg for x in ("LEFT","REMOVED","INTRUDER"))
        bc = C_WARN if is_danger else C_OK_BG
        bh2 = 34
        alpha_rect(tile, 0,th-bh2,tw,th, bc, 0.82*fade+0.05)
        cv2.line(tile,(0,th-bh2),(tw,th-bh2),C_DIM,1)
        wc  = tuple(int(v*fade) for v in C_WHITE)
        cv2.putText(tile,cs.warning_msg[:68],(88,th-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.44,wc,1,cv2.LINE_AA)

    # Uptime
    ut  = cs.uptime_str
    utw = cv2.getTextSize(ut,cv2.FONT_HERSHEY_SIMPLEX,0.30,1)[0][0]
    cv2.putText(tile,ut,(tw-utw-6,th-6),cv2.FONT_HERSHEY_SIMPLEX,0.30,C_GRAY,1,cv2.LINE_AA)

    cs.display_tile = tile
    return tile


# ══════════════════════════════════════════════════════════════════════════════
# GRID BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_grid(cams, aw, ah, anim_t):
    n = len(cams)
    if n==0: return np.zeros((ah,aw,3),dtype=np.uint8)
    if n==1: cols,rows=1,1
    elif n==2: cols,rows=2,1
    elif n<=4: cols,rows=2,2
    elif n<=6: cols,rows=3,2
    else: cols=math.ceil(math.sqrt(n)); rows=math.ceil(n/cols)
    tw=aw//cols; th=ah//rows
    tiles=[draw_tile(c,tw,th,anim_t) for c in cams]
    while len(tiles)<rows*cols: tiles.append(np.zeros((th,tw,3),dtype=np.uint8))
    return np.vstack([np.hstack(tiles[r*cols:r*cols+cols]) for r in range(rows)])


# ══════════════════════════════════════════════════════════════════════════════
# SIDE PANELS  [FIX-5] — dynamic with sparklines + hover buttons
# ══════════════════════════════════════════════════════════════════════════════

# On-screen button definitions  [(key_char, label, x_offset)]
# These are drawn in the footer area.  Clicking is not required — they're
# visual affordances showing keyboard shortcuts with a cyberpunk toggle style.
_BUTTONS = [
    ("G", "GRID"),  ("F", "FOCUS"), ("TAB", "NEXT"),
    ("L", "LEFT"),  ("R", "EVTS"), ("B", "BAR"),
    ("P", "PROF"),  ("S", "SNAP"), ("X", "CSV"), ("Q", "QUIT"),
]

def draw_left_panel(canvas, cams, x, y, w, h, anim_t):
    alpha_rect(canvas, x,y,x+w,y+h, C_PANEL, 0.96)
    # Animated side accent line
    pulse = 0.5+0.5*math.sin(anim_t*2)
    lc = tuple(int(v*pulse) for v in C_ACCENT)
    cv2.line(canvas,(x+w-1,y),(x+w-1,y+h),lc,2)

    cy = y+30
    glow_text(canvas,"[ CAMERAS ]",x+14,cy,0.50,C_ACCENT,2)
    cy+=6; cv2.line(canvas,(x+8,cy),(x+w-8,cy),C_DIM,1); cy+=14

    for cs in cams:
        if cy > y+h-140: break
        # Camera row with colored status
        dot = C_GREEN if cs.online else C_RED
        if cs.online and cs.present_pids:
            dot = C_ACCENT if int(anim_t*3)%2==0 else C_GREEN
        cv2.circle(canvas,(x+16,cy-6),5,dot,-1,cv2.LINE_AA)
        cv2.circle(canvas,(x+16,cy-6),5,tuple(v//2 for v in dot),1,cv2.LINE_AA)
        text(canvas,f"#{cs.cam_id+1} {cs.name}",x+28,cy,0.44,C_WHITE)
        cy+=16
        text(canvas,f"   {cs.fps_inst:4.1f} fps  |  det:{len(cs.latest_dets)}",
             x+10,cy,0.34,C_GRAY)
        cy+=13
        # Sparkline per camera
        draw_sparkline(canvas, x+10,cy,w-20,12, list(cs.fps_spark), C_ACCENT, C_DIM)
        cy+=16
        # Present persons list
        for pid in list(cs.present_pids):
            if cy > y+h-140: break
            nm  = cs.pid_info[pid].name[:15] if pid in cs.pid_info else "?"
            col = pid_col(pid)
            alpha_rect(canvas, x+8,cy-11,x+w-8,cy+2, col, 0.10)
            cv2.rectangle(canvas,(x+8,cy-11),(x+w-8,cy+2),tuple(v//3 for v in col),1)
            text(canvas,f"  {pid}: {nm}",x+12,cy,0.33,col)
            cy+=14
        cy+=4
        cv2.line(canvas,(x+10,cy),(x+w-10,cy),C_DIM,1); cy+=8

    # Session stats
    sy = y+h-135
    cv2.line(canvas,(x+8,sy),(x+w-8,sy),C_DIM,1); sy+=16
    glow_text(canvas,"[ SESSION ]",x+14,sy,0.42,C_ORANGE)
    sy+=16
    for cs in cams:
        text(canvas,f"  {cs.name[:12]}: {cs.persons_detected}IN/{cs.persons_left}OUT",
             x+10,sy,0.32,C_WHITE); sy+=14

    if PSUTIL_AVAILABLE:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        sy = y+h-44
        # CPU bar
        alpha_rect(canvas, x+8,sy,x+w-8,sy+12, C_DIM, 0.7)
        cpuw = int((w-16)*cpu/100)
        cpuc = C_GREEN if cpu<60 else (C_ORANGE if cpu<80 else C_RED)
        cv2.rectangle(canvas,(x+8,sy),(x+8+cpuw,sy+12),cpuc,-1)
        text(canvas,f"CPU {cpu:4.1f}%",x+10,sy+10,0.30,C_WHITE)
        sy+=16
        alpha_rect(canvas, x+8,sy,x+w-8,sy+12, C_DIM, 0.7)
        ramw = int((w-16)*ram/100)
        cv2.rectangle(canvas,(x+8,sy),(x+8+ramw,sy+12),C_BLUE,-1)
        text(canvas,f"RAM {ram:4.1f}%",x+10,sy+10,0.30,C_WHITE)

def draw_right_panel(canvas, x, y, w, h, anim_t):
    alpha_rect(canvas, x,y,x+w,y+h, C_PANEL, 0.96)
    pulse = 0.5+0.5*math.sin(anim_t*2+1.0)
    lc = tuple(int(v*pulse) for v in C_PURPLE)
    cv2.line(canvas,(x,y),(x,y+h),lc,2)

    cy = y+30
    glow_text(canvas,"[ EVENTS ]",x+14,cy,0.50,C_PURPLE,2)
    cy+=6; cv2.line(canvas,(x+6,cy),(x+w-6,cy),C_DIM,1); cy+=14

    with _event_ring_lock:
        recent = list(reversed(list(_event_ring)))

    maxc = (w-16)//7
    now = time.time()
    for short, etype, ts in recent:
        cy+=17
        if cy > y+h-10: break
        age  = now-ts
        fade = max(0.2, 1.0-age/120.0)

        if any(k in etype for k in ("REMOVED","LEFT","INTRUDER")):
            base = (100,100,220)
        elif any(k in etype for k in ("ENTERED","RETURNED","ADDED")):
            base = C_GREEN
        elif "BASELINE" in etype or "START" in etype:
            base = C_AMBER
        else:
            base = C_GRAY

        ec = tuple(int(v*fade) for v in base)
        # Highlight very recent events
        if age < 3:
            alpha_rect(canvas, x+6,cy-13,x+w-6,cy+3, base, 0.08)
        cv2.putText(canvas, short[:maxc], (x+10,cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, ec, 1, cv2.LINE_AA)

def draw_top_bar(canvas, cams, W, mode_lbl, ui_fps, anim_t):
    alpha_rect(canvas, 0,0,W,Config.TOP_H, C_BLACK, 0.94)
    # Animated bottom border
    t_off = int(anim_t*60) % W
    for seg in range(0, W, 40):
        p  = (seg+t_off)%W
        p2 = min(p+20, W)
        c  = C_ACCENT if (p//40)%3!=0 else C_BLUE
        cv2.line(canvas,(p,Config.TOP_H-1),(p2,Config.TOP_H-1),c,1)

    # Title
    glow_text(canvas,"AI SMART SURVEILLANCE",(16,36),0.78,C_ACCENT,2) if False else None
    cv2.putText(canvas,"AI SMART SURVEILLANCE",(17,36),cv2.FONT_HERSHEY_SIMPLEX,0.78,C_DIM,3)
    glow_text(canvas,"AI SMART SURVEILLANCE",16,36,0.78,C_ACCENT,2)

    # Stats
    total_prs = sum(len(c.present_pids) for c in cams)
    total_obj = sum(len(c.latest_dets)  for c in cams)
    stxt = (f"Cams:{len(cams)}   Objs:{total_obj}   Persons:{total_prs}   "
            f"DB:{len(faces_db)}   [{HW_PROFILE}]")
    cv2.putText(canvas,stxt,(340,32),cv2.FONT_HERSHEY_SIMPLEX,0.42,C_GREEN,1,cv2.LINE_AA)

    # Mode badge
    mlabel = f"[ {mode_lbl} ]"
    mlw    = cv2.getTextSize(mlabel,cv2.FONT_HERSHEY_SIMPLEX,0.52,2)[0][0]
    alpha_rect(canvas, W-mlw-80,8,W-mlw-80+mlw+10,Config.TOP_H-8, C_DIM, 0.7)
    glow_text(canvas, mlabel, W-mlw-74, 36, 0.52, C_AMBER, 2)

    # FPS + time
    perf = f"UI:{ui_fps:4.1f}fps  {datetime.now().strftime('%H:%M:%S')}"
    pw   = cv2.getTextSize(perf,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)[0][0]
    oc   = C_RED if adaptive.overloaded else C_GRAY
    cv2.putText(canvas,perf,(W-pw-6,20),cv2.FONT_HERSHEY_SIMPLEX,0.36,oc,1,cv2.LINE_AA)

    if adaptive.overloaded:
        px = W//2-160
        alpha_rect(canvas,px,4,px+320,Config.TOP_H-4,C_RED,0.18)
        pulse = 0.7+0.3*math.sin(anim_t*8)
        rc    = tuple(int(v*pulse) for v in C_RED)
        glow_text(canvas,"!! OVERLOAD PROTECTION ACTIVE !!",px+8,38,0.42,rc)

def draw_footer(canvas, W, H, active_states: dict, show_profiler: bool, anim_t):
    """[FIX-5] Dynamic footer with animated button strips."""
    y0 = H-Config.FOOTER_H
    alpha_rect(canvas, 0,y0,W,H, C_BLACK, 0.94)
    # Animated top border
    t_off = int(anim_t*40) % W
    for seg in range(0, W, 30):
        p = (seg+t_off)%W
        cv2.line(canvas,(p,y0),(min(p+15,W),y0),C_DIM,1)

    bx = 8; by = y0+7; bw = 52; bh = 26; gap = 4
    toggle_states = {
        "L": active_states.get("show_left",True),
        "R": active_states.get("show_right",True),
        "B": True,
        "P": active_states.get("show_profiler",False),
    }
    for key_char, label in _BUTTONS:
        active = toggle_states.get(key_char, None)
        # Choose button color
        if active is True:
            bg = C_ACCENT; fc = C_BLACK
        elif active is False:
            bg = C_DIM;    fc = C_GRAY
        else:
            bg = (22,28,44); fc = C_ACCENT  # action button

        alpha_rect(canvas, bx,by,bx+bw,by+bh, bg, 0.82)
        cv2.rectangle(canvas,(bx,by),(bx+bw,by+bh), C_ACCENT if active else C_DIM, 1)

        # Key label small
        kw = cv2.getTextSize(key_char,cv2.FONT_HERSHEY_SIMPLEX,0.28,1)[0][0]
        alpha_rect(canvas, bx+2,by+2,bx+kw+8,by+13, C_DIM if active else C_ACCENT, 0.6)
        cv2.putText(canvas,key_char,(bx+4,by+11),cv2.FONT_HERSHEY_SIMPLEX,0.28,
                    C_WHITE if active else C_BLACK,1,cv2.LINE_AA)
        # Main label
        lw = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.30,1)[0][0]
        cv2.putText(canvas,label,(bx+(bw-lw)//2,by+bh-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.30,fc,1,cv2.LINE_AA)
        bx += bw+gap

def draw_profiler(canvas, cams, W, H, ui_fps, anim_t):
    oy=Config.TOP_H+6; ox=10; bw=110
    cpu = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 0
    ram = psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0
    alpha_rect(canvas,ox,oy,ox+520,oy+18+(len(cams))*18, C_BLACK, 0.75)
    glow_text(canvas,
              f"PROFILER | CPU:{cpu:.0f}% RAM:{ram:.0f}% | det:{adaptive.det_w}x{adaptive.det_h} skip:{adaptive.skip_n}",
              ox+4,oy+14,0.36,C_AMBER)
    oy+=22
    for cs in cams:
        ratio = min(cs.fps_inst/max(1,Config.TARGET_FPS),1.0)
        fill  = int(bw*ratio)
        alpha_rect(canvas,ox,oy,ox+bw,oy+12,C_DIM,0.8)
        bc = C_GREEN if ratio>0.6 else (C_ORANGE if ratio>0.3 else C_RED)
        cv2.rectangle(canvas,(ox,oy),(ox+fill,oy+12),bc,-1)
        text(canvas,f" {cs.name[:10]} {cs.fps_inst:.1f}fps",
             ox+bw+6,oy+10,0.32,C_WHITE)
        oy+=17


# ══════════════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════════════

active_cameras: List[CameraState] = []
for _i, _cfg in enumerate(Config.CAMERA_CONFIGS):
    if _cfg.get("enabled", True):
        _cs = CameraState(_i, _cfg)
        active_cameras.append(_cs)

for _cs in active_cameras:
    _t = threading.Thread(target=camera_thread, args=(_cs,),
                          daemon=True, name=f"Cam-{_cs.name}")
    _t.start()
    app_log.info(f"Thread started: {_cs.name}")

log_event("SYSTEM_START", detail=f"{len(active_cameras)} cams profile={HW_PROFILE}")
notif_queue.send_message(
    f"\U0001F7E2 <b>SURVEILLANCE STARTED</b>\n"
    f"\U0001F4F7 Cameras: {', '.join(c.name for c in active_cameras)}\n"
    f"\U0001F4BB Profile: {HW_PROFILE}  GPU: {CUDA_DEVICE}\n"
    f"\u23F0 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    event_type="SYSTEM")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DISPLAY LOOP  [FIX-4] — UI fully decoupled from detection
# ══════════════════════════════════════════════════════════════════════════════

W, H   = Config.WINDOW_W, Config.WINDOW_H
canvas = np.zeros((H,W,3), dtype=np.uint8)  # pre-allocated, never reallocated

show_left     = True
show_right    = True
show_footer   = True
show_profiler = False
view_mode     = "grid"
cam_idx       = 0

_db_save_t  = time.time()
_gc1_t      = time.time()
_fps_t      = time.perf_counter()
_fps_cnt    = 0
_ui_fps     = 0.0
_anim_start = time.time()

cv2.namedWindow("AI SURVEILLANCE", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI SURVEILLANCE", W, H)
cv2.setWindowProperty("AI SURVEILLANCE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    loop_t = time.perf_counter()
    now    = time.time()
    anim_t = now - _anim_start   # ever-increasing time for animations

    # Maintenance
    if now-_db_save_t > Config.DB_SAVE_SECS:
        _save_db(faces_db); _db_save_t=now
    if now-_gc1_t > Config.GC_GEN1_SECS:
        gc.collect(1); _gc1_t=now
    adaptive.update()

    # Layout
    lw = Config.SIDE_W   if show_left   else 0
    rw = Config.EVENT_W  if show_right  else 0
    fh = Config.FOOTER_H if show_footer else 0
    ax = lw; ay = Config.TOP_H
    aw = W-lw-rw; ah = H-Config.TOP_H-fh

    # Clear camera zone only
    canvas[ay:ay+ah, ax:ax+aw] = C_BG

    # Camera feeds
    if view_mode=="grid":
        grid  = build_grid(active_cameras, aw, ah, anim_t)
        h_fit = min(ah, grid.shape[0]); w_fit = min(aw, grid.shape[1])
        np.copyto(canvas[ay:ay+h_fit, ax:ax+w_fit], grid[:h_fit,:w_fit])
    else:
        cs   = active_cameras[cam_idx % max(1,len(active_cameras))]
        tile = draw_tile(cs, aw, ah, anim_t)
        np.copyto(canvas[ay:ay+ah, ax:ax+aw], tile)

    # Chrome
    mode_lbl = "GRID" if view_mode=="grid" else f"CAM {cam_idx+1}"
    draw_top_bar(canvas, active_cameras, W, mode_lbl, _ui_fps, anim_t)

    if show_left:
        draw_left_panel(canvas, active_cameras, 0, Config.TOP_H,
                        Config.SIDE_W, H-Config.TOP_H-fh, anim_t)
    if show_right:
        draw_right_panel(canvas, W-Config.EVENT_W, Config.TOP_H,
                         Config.EVENT_W, H-Config.TOP_H-fh, anim_t)
    if show_footer:
        states = {"show_left":show_left,"show_right":show_right,
                  "show_profiler":show_profiler}
        draw_footer(canvas, W, H, states, show_profiler, anim_t)
    if show_profiler:
        draw_profiler(canvas, active_cameras, W, H, _ui_fps, anim_t)

    # UI FPS
    _fps_cnt += 1
    eu = time.perf_counter()-_fps_t
    if eu >= 1.0:
        _ui_fps=_fps_cnt/eu; _fps_cnt=0; _fps_t=time.perf_counter()

    cv2.imshow("AI SURVEILLANCE", canvas)

    # Adaptive sleep — give OS exactly the remaining budget
    used_ms = int((time.perf_counter()-loop_t)*1000)
    wait_ms = max(1, adaptive.frame_ms-used_ms)
    key     = cv2.waitKey(wait_ms) & 0xFF

    if   key==ord('q'): break
    elif key==ord('g'): view_mode="grid"
    elif key==ord('f'): view_mode="focus"
    elif key==9:        cam_idx=(cam_idx+1)%max(1,len(active_cameras)); view_mode="focus"
    elif key==ord('l'): show_left=not show_left
    elif key==ord('r'): show_right=not show_right
    elif key==ord('b'): show_footer=not show_footer
    elif key==ord('p'): show_profiler=not show_profiler
    elif key==ord('d'):
        for c in active_cameras: c.tile_dirty=True
    elif key==ord('s'):
        log_event("EVIDENCE", detail="manual snapshot")
        notif_queue.send_message("\U0001F4F8 <b>EVIDENCE</b> — sending snapshots",
                                 event_type="SYSTEM")
        for c in active_cameras:
            if os.path.exists(c.before_img): notif_queue.send_photo(c.before_img,f"BEFORE {c.name}")
            if os.path.exists(c.after_img):  notif_queue.send_photo(c.after_img, f"AFTER {c.name}")
    elif key==ord('x'): export_csv(); log_event("CSV_EXPORT")
    elif ord('1')<=key<=ord('9'):
        idx=key-ord('1')
        if idx<len(active_cameras): cam_idx=idx; view_mode="focus"


# ══════════════════════════════════════════════════════════════════════════════
# SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

_save_db(faces_db)
log_event("SYSTEM_STOP", detail="session ended")
export_csv()

notif_queue.send_message(
    f"\U0001F534 <b>SURVEILLANCE STOPPED</b>\n\u23F0 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    event_type="SYSTEM")

_lq.put(None); _log_th.join(timeout=5)
for _cs in active_cameras: _cs.release()
face_pool.shutdown(wait=False)
cv2.destroyAllWindows()
gc.collect()
app_log.info("Clean shutdown.")
print("System closed.")
