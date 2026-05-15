# =====================================================
# AI SMART SURVEILLANCE SYSTEM — STABLE VERSION
# YOLO Track ID → Face PID binding | Grace buffer
# No more flickering LEFT/RETURNED
# =====================================================

from ultralytics import YOLO
import cv2
import time
import requests
import threading
import os
import json
import platform
import face_recognition
import numpy as np

if platform.system() == "Windows":
    import winsound

from collections import Counter
from datetime import datetime

# =====================================================
# DIRECTORY SETUP
# =====================================================

for d in ["logs", "faces/known", "faces/unknown", "faces/captured"]:
    os.makedirs(d, exist_ok=True)

# =====================================================
# PATHS
# =====================================================

LOG_FILE       = "logs/surveillance_log.txt"
EVENT_LOG_FILE = "logs/events_table.txt"
FACES_DB_FILE  = "logs/faces_db.json"
BEFORE_IMG     = "logs/before.jpg"
AFTER_IMG      = "logs/after.jpg"

# =====================================================
# TELEGRAM SETTINGS
# =====================================================

BOT_TOKEN = "8938780809:AAHzpgv_fbfbmXJ9x_ui44LY83CWnTWfKPo"
CHAT_ID   = "8076971661"

# =====================================================
# SETTINGS
# =====================================================

MODEL_NAME         = "yolov8n.pt"
FRAME_WIDTH        = 960
FRAME_HEIGHT       = 540
DETECTION_WIDTH    = 640
DETECTION_HEIGHT   = 360
CONFIDENCE         = 0.55
PROCESS_EVERY_N   = 4          # run YOLO every N frames

# ── Face recognition ──────────────────────────────
FACE_MATCH_THRESH  = 0.52      # lower = stricter (good range 0.45–0.55)
KNOWN_FACES_DIR    = "faces/known"

# ── Departure detection ───────────────────────────
# A person is only declared LEFT after being absent
# for this many CONSECUTIVE detection cycles.
# At PROCESS_EVERY_N=4 and 30fps → 1 cycle ≈ 0.13s
# 40 cycles ≈ 5 seconds grace before firing LEFT
ABSENT_CYCLES_THRESHOLD = 40

# ── Face re-identification ────────────────────────
# How many cycles to wait before trying face-recog
# again on an already-bound track (saves CPU)
FACE_RECHECK_CYCLES = 30

# =====================================================
# COLORS  (BGR)
# =====================================================

BLACK  = (15,  15,  15)
DARK   = (25,  25,  25)
CYAN   = (255, 255,  0)
GREEN  = (0,   255, 100)
WHITE  = (255, 255, 255)
RED    = (0,     0, 255)
BLUE   = (255, 120,  0)
ORANGE = (0,   165, 255)

_COLOR_POOL = [
    (0, 255, 200), (255, 200, 0), (200, 0, 255),
    (0, 200, 255), (255, 80,  0), (80, 255, 0),
    (0, 80,  255), (255, 0,  150),(0, 255, 80),
    (150, 0, 255), (255, 150, 0), (0, 150, 255),
]
_pid_color_map = {}

def pid_color(pid):
    if pid not in _pid_color_map:
        _pid_color_map[pid] = _COLOR_POOL[len(_pid_color_map) % len(_COLOR_POOL)]
    return _pid_color_map[pid]

# =====================================================
# FACES DATABASE
# =====================================================

def load_faces_db():
    if os.path.exists(FACES_DB_FILE):
        with open(FACES_DB_FILE, "r") as f:
            db = json.load(f)
        for pid, data in db.items():
            data["encoding"] = np.array(data["encoding"])
        return db
    return {}

def save_faces_db(db):
    out = {}
    for pid, data in db.items():
        out[pid] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in data.items()
        }
    with open(FACES_DB_FILE, "w") as f:
        json.dump(out, f, indent=2)

faces_db = load_faces_db()
_next_pid_n = 1
for pid in faces_db:
    try:
        n = int(pid[1:])
        if n >= _next_pid_n:
            _next_pid_n = n + 1
    except:
        pass

def get_next_pid():
    global _next_pid_n
    pid = f"P{_next_pid_n}"
    _next_pid_n += 1
    return pid

# =====================================================
# PRELOAD KNOWN FACES
# =====================================================

def preload_known_faces():
    known_by_name = {
        data["name"]: pid
        for pid, data in faces_db.items()
        if data.get("known", False)
    }
    loaded = []
    for file in os.listdir(KNOWN_FACES_DIR):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        name  = os.path.splitext(file)[0]
        path  = os.path.join(KNOWN_FACES_DIR, file)
        image = face_recognition.load_image_file(path)
        encs  = face_recognition.face_encodings(image)
        if not encs:
            print(f"  WARNING: no face found in {file}")
            continue
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if name in known_by_name:
            pid = known_by_name[name]
            faces_db[pid]["encoding"] = encs[0]   # refresh encoding
        else:
            pid = get_next_pid()
            faces_db[pid] = {
                "name":        name,
                "encoding":    encs[0],
                "first_seen":  now,
                "last_seen":   now,
                "visit_count": 0,
                "known":       True,
                "photo":       path,
            }
        loaded.append(name)
    save_faces_db(faces_db)
    print(f"Known faces loaded: {loaded}")

preload_known_faces()

# =====================================================
# FACE MATCHING
# =====================================================

def match_face(encoding):
    """Returns (pid, name, is_new)."""
    if not faces_db:
        return _register_face(encoding)

    all_encs  = [d["encoding"] for d in faces_db.values()]
    all_pids  = list(faces_db.keys())
    distances = face_recognition.face_distance(all_encs, encoding)
    best_i    = int(np.argmin(distances))
    best_d    = distances[best_i]

    if best_d <= FACE_MATCH_THRESH:
        pid  = all_pids[best_i]
        name = faces_db[pid]["name"]
        return pid, name, False
    return _register_face(encoding)

def _register_face(encoding):
    pid  = get_next_pid()
    name = f"Intruder-{pid}"
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    faces_db[pid] = {
        "name":        name,
        "encoding":    encoding,
        "first_seen":  now,
        "last_seen":   now,
        "visit_count": 0,
        "known":       False,
        "photo":       None,
    }
    save_faces_db(faces_db)
    return pid, name, True

# =====================================================
# STRUCTURED LOG
# =====================================================

COL_W = [21, 16, 10, 22, 32]

def _row(*cols):
    parts = []
    for i, c in enumerate(cols):
        w = COL_W[i] if i < len(COL_W) else 32
        parts.append(str(c)[:w].ljust(w))
    return "| " + " | ".join(parts) + " |"

def _div():
    return "+-" + "-+-".join("-" * w for w in COL_W) + "-+"

def init_event_log():
    if not os.path.exists(EVENT_LOG_FILE):
        with open(EVENT_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=" * 114 + "\n")
            f.write(" AI SMART SURVEILLANCE SYSTEM — EVENT LOG".center(114) + "\n")
            f.write(f" Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(114) + "\n")
            f.write("=" * 114 + "\n\n")
            f.write(_div() + "\n")
            f.write(_row("TIMESTAMP", "EVENT", "OBJECT ID", "NAME", "DETAIL") + "\n")
            f.write(_div() + "\n")

init_event_log()
history_log = []

def log_event(event_type, obj_id="—", name="—", detail=""):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = _row(ts, event_type, obj_id, name, detail)
    try:
        with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(row + "\n")
    except Exception as e:
        print("LOG ERROR:", e)

    short = f"[{ts}] {event_type:<14} | {obj_id:<6} | {name:<18} | {detail}"
    history_log.append(short)
    print(short)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(short + "\n")
    except:
        pass

# =====================================================
# TELEGRAM
# =====================================================

def _send_msg(text):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text}, timeout=5)
    except Exception as e:
        print("TELEGRAM:", e)

def _send_photo(path, caption):
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as p:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                files={"photo": p},
                data={"chat_id": CHAT_ID, "caption": caption}, timeout=10)
    except Exception as e:
        print("PHOTO:", e)

def async_alert(msg):
    threading.Thread(target=_send_msg,   args=(msg,),       daemon=True).start()

def async_photo(path, cap):
    threading.Thread(target=_send_photo, args=(path, cap),  daemon=True).start()

def play_alarm():
    try:
        if platform.system() == "Windows":
            winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)
    except:
        pass

# =====================================================
# LOAD MODEL + CAMERA
# =====================================================

print("Loading YOLO...")
model = YOLO(MODEL_NAME)
print("YOLO loaded.")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
if not cap.isOpened():
    print("Cannot open camera"); exit()

# =====================================================
# TRACKING STATE
# =====================================================

frame_count       = 0
latest_detections = []
object_counts     = Counter()

show_left_panel   = True
show_right_panel  = True
show_footer       = True
show_boxes        = True

startup_time      = time.time()
baseline_saved    = False
last_seen_counter = Counter()
confirmed_counts  = Counter()
obj_missing_since = {}
obj_added_since   = {}
active_warning    = ""
warning_time      = 0.0
_last_db_save     = time.time()

# ── Person tracking ───────────────────────────────
#
# track_to_pid  : { yolo_track_id (int) -> pid (str) }
#   Once a YOLO track ID is bound to a pid, we reuse
#   that binding every frame — no re-running face-recog
#   unless the track is new or we schedule a recheck.
#
# pid_info      : { pid -> { name, absent_cycles,
#                            present, last_box,
#                            face_check_countdown } }
#
# present_pids  : set of pids currently in frame

track_to_pid          = {}   # yolo_track_id -> pid
pid_info              = {}   # pid -> info dict
present_pids          = set()

def _ensure_pid_info(pid, name):
    if pid not in pid_info:
        pid_info[pid] = {
            "name":                name,
            "absent_cycles":       0,
            "present":             False,
            "last_box":            None,
            "face_check_countdown": 0,
        }
    else:
        pid_info[pid]["name"] = name   # keep name updated

# =====================================================
# HELPER UTILS
# =====================================================

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def transparent_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)

def draw_label(img, text, x1, y1, color):
    tw = len(text) * 10 + 12
    cv2.rectangle(img, (x1, y1 - 30), (x1 + tw, y1), color, -1)
    cv2.putText(img, text, (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2)

# =====================================================
# FACE RECOGNITION FOR ONE PERSON BOX
# =====================================================

def try_recognize_face(frame, x1, y1, x2, y2):
    """Return (pid, name) or (None, None) if no face found."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None
    try:
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return None, None
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return None, None
        pid, name, is_new = match_face(encs[0])
        return pid, name
    except Exception as e:
        print("FACE ERROR:", e)
        return None, None

# =====================================================
# PERSON ENTER / RETURN HANDLER
# =====================================================

def on_person_arrived(pid, name, frame, x1, y1, x2, y2):
    now     = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    db      = faces_db[pid]
    db["last_seen"]   = now_str
    db["visit_count"] = db.get("visit_count", 0) + 1

    visit_n = db["visit_count"]
    is_new  = visit_n == 1

    event = "ENTERED" if is_new else "RETURNED"
    emoji = "🟢" if is_new else "🔄"
    word  = "NEW PERSON DETECTED" if is_new else "PERSON RETURNED"

    log_event(event, pid, name, f"Visit #{visit_n}")
    async_alert(
        f"{emoji} {word}\n\n"
        f"ID   : {pid}\n"
        f"Name : {name}\n"
        f"Visit: #{visit_n}\n"
        f"Time : {now_str}"
    )

    # Save face capture
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        cap_path = f"faces/captured/{pid}_{int(time.time())}.jpg"
        cv2.imwrite(cap_path, crop)
        if not db.get("known") and db.get("photo") is None:
            db["photo"] = cap_path

    save_faces_db(faces_db)

def on_person_left(pid, frame):
    name    = pid_info[pid]["name"]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.imwrite(AFTER_IMG, frame)
    log_event("LEFT", pid, name, f"Last seen {now_str}")
    async_alert(
        f"⚠️ PERSON LEFT\n\n"
        f"ID   : {pid}\n"
        f"Name : {name}\n"
        f"Time : {now_str}"
    )
    threading.Thread(target=play_alarm, daemon=True).start()

# =====================================================
# WINDOW
# =====================================================

cv2.namedWindow("AI SURVEILLANCE", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI SURVEILLANCE",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

log_event("SYSTEM START", "—", "—", "Surveillance initialized")

# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1
    display = frame.copy()

    # ── Periodic DB save ──────────────────────────
    if time.time() - _last_db_save > 30:
        save_faces_db(faces_db)
        _last_db_save = time.time()

    # ── Baseline snapshot ─────────────────────────
    if not baseline_saved and time.time() - startup_time > 5:
        cv2.imwrite(BEFORE_IMG, frame)
        baseline_saved = True
        confirmed_counts = last_seen_counter.copy()
        async_alert("✅ SURVEILLANCE ACTIVE\nBaseline saved.")
        log_event("BASELINE", "—", "—", "Snapshot saved")

    # =================================================
    # YOLO DETECTION CYCLE
    # =================================================

    if frame_count % PROCESS_EVERY_N == 0:

        small   = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
        results = model.track(source=small, conf=CONFIDENCE,
                               persist=True, verbose=False)
        boxes   = results[0].boxes

        sx = FRAME_WIDTH  / DETECTION_WIDTH
        sy = FRAME_HEIGHT / DETECTION_HEIGHT

        latest_detections  = []
        current_objects    = []
        track_ids_seen     = set()   # yolo track IDs visible this cycle

        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONFIDENCE:
                continue

            cls   = int(box.cls[0])
            label = model.names[cls]

            bx1, by1, bx2, by2 = box.xyxy[0]
            x1 = clamp(int(bx1 * sx), 0, FRAME_WIDTH  - 1)
            y1 = clamp(int(by1 * sy), 0, FRAME_HEIGHT - 1)
            x2 = clamp(int(bx2 * sx), 0, FRAME_WIDTH  - 1)
            y2 = clamp(int(by2 * sy), 0, FRAME_HEIGHT - 1)

            current_objects.append(label)

            pid       = None
            disp_name = label

            # ──────────────────────────────────────
            # PERSON: bind YOLO track ID → face pid
            # ──────────────────────────────────────
            if label == "person":

                # Get YOLO's own track ID for this box
                yolo_tid = None
                if box.id is not None:
                    try:
                        yolo_tid = int(box.id[0])
                    except:
                        pass

                if yolo_tid is not None:
                    track_ids_seen.add(yolo_tid)

                    if yolo_tid in track_to_pid:
                        # ── Already bound: reuse existing pid ──
                        pid  = track_to_pid[yolo_tid]
                        name = pid_info[pid]["name"]

                        # Periodically re-run face-recog to confirm
                        info = pid_info[pid]
                        info["face_check_countdown"] -= 1
                        if info["face_check_countdown"] <= 0:
                            np_pid, np_name = try_recognize_face(
                                frame, x1, y1, x2, y2)
                            info["face_check_countdown"] = FACE_RECHECK_CYCLES
                            if np_pid and np_pid != pid:
                                # face says different person — rebind
                                track_to_pid[yolo_tid] = np_pid
                                pid  = np_pid
                                name = np_name
                                _ensure_pid_info(pid, name)

                    else:
                        # ── New track: run face recognition now ──
                        np_pid, np_name = try_recognize_face(
                            frame, x1, y1, x2, y2)

                        if np_pid:
                            pid  = np_pid
                            name = np_name
                            track_to_pid[yolo_tid] = pid
                            _ensure_pid_info(pid, name)
                        # else: face not detected yet — leave unbound
                        # (will retry next cycle since track_id not in map)

                    if pid:
                        info = pid_info[pid]
                        info["last_box"]       = (x1, y1, x2, y2)
                        info["absent_cycles"]  = 0   # reset absence counter

                        if not info["present"]:
                            # Person just arrived / returned
                            info["present"] = True
                            present_pids.add(pid)
                            on_person_arrived(pid, info["name"],
                                              frame, x1, y1, x2, y2)

                        disp_name = f"{pid} | {info['name']}"

            latest_detections.append({
                "label":     label,
                "conf":      conf,
                "box":       (x1, y1, x2, y2),
                "disp_name": disp_name,
                "pid":       pid,
            })

        object_counts     = Counter(current_objects)
        current_counter   = Counter(current_objects)
        last_seen_counter = current_counter.copy()

        # ──────────────────────────────────────────
        # DEPARTURE CHECK
        # Increment absent_cycles for every pid whose
        # YOLO track is no longer visible this cycle.
        # Fire LEFT only after ABSENT_CYCLES_THRESHOLD.
        # ──────────────────────────────────────────

        # Find which pids had their track disappear
        pids_active_this_cycle = set()
        for tid in track_ids_seen:
            if tid in track_to_pid:
                pids_active_this_cycle.add(track_to_pid[tid])

        for pid in list(present_pids):
            if pid not in pids_active_this_cycle:
                pid_info[pid]["absent_cycles"] += 1
                if pid_info[pid]["absent_cycles"] >= ABSENT_CYCLES_THRESHOLD:
                    # Confirmed departure
                    present_pids.discard(pid)
                    pid_info[pid]["present"]       = False
                    pid_info[pid]["absent_cycles"] = 0
                    # Clean up stale track binding so a fresh
                    # detection later can rebind correctly
                    stale_tids = [t for t, p in track_to_pid.items()
                                  if p == pid]
                    for t in stale_tids:
                        del track_to_pid[t]
                    on_person_left(pid, frame)
                    active_warning = f"LEFT: {pid_info[pid]['name'].upper()}"
                    warning_time   = time.time()

        # ──────────────────────────────────────────
        # NON-PERSON OBJECT TRACKING
        # ──────────────────────────────────────────

        if baseline_saved:
            now = time.time()

            for obj, conf_c in list(confirmed_counts.items()):
                cur_c = current_counter.get(obj, 0)
                if cur_c < conf_c:
                    if obj not in obj_missing_since:
                        obj_missing_since[obj] = now
                    if now - obj_missing_since[obj] >= 1.5:
                        removed_n = conf_c - cur_c
                        confirmed_counts[obj] = cur_c
                        if cur_c == 0:
                            del confirmed_counts[obj]
                        del obj_missing_since[obj]
                        lbl = f"{obj} x{removed_n}" if removed_n > 1 else obj
                        log_event("OBJ REMOVED", "—", lbl,
                                  f"{conf_c} → {cur_c}")
                        cv2.imwrite(AFTER_IMG, frame)
                        threading.Thread(target=play_alarm,
                                         daemon=True).start()
                        async_alert(
                            f"⚠️ OBJECT REMOVED\nObject: {lbl}\n"
                            f"Was: {conf_c} → Now: {cur_c}\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}")
                        active_warning = f"REMOVED: {lbl.upper()}"
                        warning_time   = now
                else:
                    obj_missing_since.pop(obj, None)

            for obj, cur_c in current_counter.items():
                conf_c = confirmed_counts.get(obj, 0)
                if cur_c > conf_c:
                    if obj not in obj_added_since:
                        obj_added_since[obj] = now
                    if now - obj_added_since[obj] >= 1.5:
                        added_n = cur_c - conf_c
                        confirmed_counts[obj] = cur_c
                        del obj_added_since[obj]
                        lbl = f"{obj} x{cur_c}" if added_n > 1 or conf_c > 0 else obj
                        log_event("OBJ ADDED", "—", lbl,
                                  f"{conf_c} → {cur_c}")
                        async_alert(
                            f"🟢 OBJECT ADDED\nObject: {lbl}\n"
                            f"Was: {conf_c} → Now: {cur_c}\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}")
                        active_warning = f"ADDED: {lbl.upper()}"
                        warning_time   = now
                else:
                    obj_added_since.pop(obj, None)

    # =================================================
    # DRAW
    # =================================================

    if show_boxes:
        for d in latest_detections:
            x1, y1, x2, y2 = d["box"]
            pid   = d.get("pid")
            color = pid_color(pid) if pid else CYAN
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
            draw_label(display,
                       f"{d['disp_name']}  {d['conf']:.2f}",
                       x1, y1, color)

    H, W = display.shape[:2]

    # ── Top bar ───────────────────────────────────
    transparent_rect(display, 0, 0, W, 56, BLACK, 0.88)
    cv2.putText(display, "AI SMART SURVEILLANCE",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.88, CYAN, 2)
    cv2.putText(display,
                f"Objects: {len(latest_detections)}  |  "
                f"Persons: {len(present_pids)}  |  "
                f"DB: {len(faces_db)} faces",
                (360, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, GREEN, 1)
    st_txt   = "READY" if baseline_saved else "SCANNING..."
    st_color = GREEN   if baseline_saved else RED
    cv2.putText(display, st_txt, (W - 140, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, st_color, 2)

    # ── Warning banner ────────────────────────────
    if time.time() - warning_time < 3:
        bc = RED if "REMOVED" in active_warning or "LEFT" in active_warning else GREEN
        cv2.rectangle(display, (W//2-290, 60), (W//2+290, 108), bc, -1)
        cv2.putText(display, active_warning,
                    (W//2-270, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 3)

    # ── Left panel ────────────────────────────────
    if show_left_panel:
        transparent_rect(display, 0, 56, 240, H, DARK, 0.78)
        cv2.putText(display, "LIVE OBJECTS",
                    (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        y = 124
        for obj, cnt in object_counts.items():
            cv2.putText(display, f"  {obj}: {cnt}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 1)
            y += 34
        y += 10
        cv2.putText(display, "PRESENT PERSONS",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ORANGE, 2)
        y += 28
        for pid in present_pids:
            name  = pid_info[pid]["name"]
            color = pid_color(pid)
            cv2.putText(display, f"  {pid}: {name[:16]}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 28

    # ── Right panel ───────────────────────────────
    if show_right_panel:
        transparent_rect(display, W-370, 56, W, H, DARK, 0.78)
        cv2.putText(display, "EVENTS",
                    (W-350, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
        y = 124
        for ev in reversed(history_log[-20:]):
            if "REMOVED" in ev or "LEFT" in ev:
                ec = RED
            elif "ENTERED" in ev or "RETURNED" in ev or "ADDED" in ev:
                ec = GREEN
            else:
                ec = WHITE
            cv2.putText(display, ev[:56],
                        (W-360, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, ec, 1)
            y += 25

    # ── Footer ────────────────────────────────────
    if show_footer:
        transparent_rect(display, 0, H-42, W, H, BLACK, 0.88)
        cv2.putText(display,
                    "L:Left Panel  R:Events  D:Boxes  B:Footer  S:Evidence  Q:Quit",
                    (16, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)

    cv2.imshow("AI SURVEILLANCE", display)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('l'): show_left_panel  = not show_left_panel
    elif key == ord('r'): show_right_panel = not show_right_panel
    elif key == ord('d'): show_boxes       = not show_boxes
    elif key == ord('b'): show_footer      = not show_footer
    elif key == ord('s'):
        log_event("EVIDENCE", "—", "—", "Manual snapshot sent")
        async_alert("📸 SENDING EVIDENCE")
        if os.path.exists(BEFORE_IMG): async_photo(BEFORE_IMG, "📷 BEFORE")
        if os.path.exists(AFTER_IMG):  async_photo(AFTER_IMG,  "📷 AFTER")

# =====================================================
# CLEANUP
# =====================================================

save_faces_db(faces_db)
log_event("SYSTEM STOP", "—", "—", "Session ended")
with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
    f.write(_div() + "\n\n")
    f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

cap.release()
cv2.destroyAllWindows()
print("System Closed")
