# =====================================================
# AI SMART SURVEILLANCE SYSTEM — FINAL
# Stable Face IDs | Persistent DB | Structured Logs
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

BOT_TOKEN = "Bot_Tokem"
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
PROCESS_EVERY_N   = 4
CONFIRMATION_TIME  = 1.5
FACE_MATCH_THRESH  = 0.50   # lower = stricter match (0.0–1.0)
FACE_ABSENT_SECS   = 4.0    # seconds before "left" is confirmed
KNOWN_FACES_DIR    = "faces/known"

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
PURPLE = (200,  50, 200)

# =====================================================
# FACES DATABASE
# Stores: { "P1": { "name": "Alice", "encoding": [...],
#           "first_seen": "...", "last_seen": "...",
#           "visit_count": 1, "known": true/false } }
# =====================================================

def load_faces_db():
    if os.path.exists(FACES_DB_FILE):
        with open(FACES_DB_FILE, "r") as f:
            db = json.load(f)
        # Restore encodings as numpy arrays
        for pid, data in db.items():
            data["encoding"] = np.array(data["encoding"])
        return db
    return {}

def save_faces_db(db):
    serializable = {}
    for pid, data in db.items():
        serializable[pid] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in data.items()
        }
    with open(FACES_DB_FILE, "w") as f:
        json.dump(serializable, f, indent=2)

faces_db = load_faces_db()   # pid -> data dict
next_pid_number = 1
for pid in faces_db:
    try:
        n = int(pid[1:])
        if n >= next_pid_number:
            next_pid_number = n + 1
    except:
        pass

def get_next_pid():
    global next_pid_number
    pid = f"P{next_pid_number}"
    next_pid_number += 1
    return pid

# =====================================================
# PRELOAD KNOWN FACES INTO DB
# Files in faces/known/ named like "Alice.jpg"
# These always get a stable known=True entry
# =====================================================

def preload_known_faces():
    global faces_db
    known_names_in_db = {
        data["name"]: pid
        for pid, data in faces_db.items()
        if data.get("known", False)
    }
    for file in os.listdir(KNOWN_FACES_DIR):
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        name = os.path.splitext(file)[0]
        image = face_recognition.load_image_file(
            os.path.join(KNOWN_FACES_DIR, file)
        )
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            continue
        if name in known_names_in_db:
            # Update encoding in case photo changed
            pid = known_names_in_db[name]
            faces_db[pid]["encoding"] = encodings[0]
        else:
            pid = get_next_pid()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            faces_db[pid] = {
                "name":        name,
                "encoding":    encodings[0],
                "first_seen":  now,
                "last_seen":   now,
                "visit_count": 0,
                "known":       True,
                "photo":       os.path.join(KNOWN_FACES_DIR, file),
            }
    save_faces_db(faces_db)
    print(f"Known faces loaded: {[d['name'] for d in faces_db.values() if d.get('known')]}")

preload_known_faces()

# =====================================================
# FACE MATCHING
# Returns (pid, name, is_new) for a given encoding
# =====================================================

def match_face(encoding):
    """Compare encoding against all DB entries.
    Returns (pid, name, is_new_person)."""
    if not faces_db:
        return _register_new_face(encoding)

    all_encodings = [data["encoding"] for data in faces_db.values()]
    all_pids      = list(faces_db.keys())

    distances = face_recognition.face_distance(all_encodings, encoding)
    best_idx  = int(np.argmin(distances))
    best_dist = distances[best_idx]

    if best_dist <= FACE_MATCH_THRESH:
        pid  = all_pids[best_idx]
        name = faces_db[pid]["name"]
        return pid, name, False
    else:
        return _register_new_face(encoding)

def _register_new_face(encoding):
    pid  = get_next_pid()
    name = f"Intruder-{pid}"
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    faces_db[pid] = {
        "name":        name,
        "encoding":    encoding,
        "first_seen":  now,
        "last_seen":   now,
        "visit_count": 1,
        "known":       False,
        "photo":       None,
    }
    save_faces_db(faces_db)
    return pid, name, True

# =====================================================
# STRUCTURED LOG  (table format)
# Columns: Timestamp | Event | Object-ID | Name | Detail
# =====================================================

COL_W = [21, 16, 10, 20, 30]   # column widths

def _table_row(*cols):
    parts = []
    for i, col in enumerate(cols):
        w   = COL_W[i] if i < len(COL_W) else 30
        txt = str(col)[:w]
        parts.append(txt.ljust(w))
    return "| " + " | ".join(parts) + " |"

def _table_divider():
    segs = ["-" * w for w in COL_W]
    return "+-" + "-+-".join(segs) + "-+"

TABLE_HEADER = (
    _table_divider() + "\n" +
    _table_row("TIMESTAMP", "EVENT", "OBJECT ID", "NAME", "DETAIL") + "\n" +
    _table_divider()
)

def init_event_log():
    if not os.path.exists(EVENT_LOG_FILE):
        with open(EVENT_LOG_FILE, "w", encoding="utf-8") as f:
            f.write("=" * 108 + "\n")
            f.write(" AI SMART SURVEILLANCE SYSTEM — EVENT LOG".center(108) + "\n")
            f.write(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(108) + "\n")
            f.write("=" * 108 + "\n\n")
            f.write(TABLE_HEADER + "\n")

init_event_log()

def log_event(event_type, obj_id="—", name="—", detail=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row       = _table_row(timestamp, event_type, obj_id, name, detail)

    # Append to table log
    try:
        with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(row + "\n")
    except Exception as e:
        print("LOG ERROR:", e)

    # Also append to simple log
    full = f"[{timestamp}] {event_type:<16} | {obj_id:<10} | {name:<20} | {detail}"
    history_log.append(full)
    print(full)

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(full + "\n")
    except:
        pass

# =====================================================
# TELEGRAM
# =====================================================

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=5)
    except Exception as e:
        print("TELEGRAM ERROR:", e)

def send_photo_telegram(path, caption=""):
    if not os.path.exists(path):
        return
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        with open(path, "rb") as p:
            requests.post(url, files={"photo": p},
                          data={"chat_id": CHAT_ID, "caption": caption}, timeout=10)
    except Exception as e:
        print("PHOTO ERROR:", e)

def async_alert(msg):
    threading.Thread(target=send_telegram, args=(msg,), daemon=True).start()

def async_photo(path, caption):
    threading.Thread(target=send_photo_telegram, args=(path, caption), daemon=True).start()

def play_alarm():
    try:
        if platform.system() == "Windows":
            winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)
    except:
        pass

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading YOLO...")
model = YOLO(MODEL_NAME)
print("YOLO Loaded")

# =====================================================
# CAMERA
# =====================================================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# =====================================================
# STATE VARIABLES
# =====================================================

frame_count      = 0
latest_detections = []
object_counts    = Counter()
history_log      = []

show_left_panel  = True
show_right_panel = True
show_footer      = True
show_boxes       = True

startup_time     = time.time()
baseline_saved   = False

# Object (non-person) tracking
last_seen_counter   = Counter()
confirmed_counts    = Counter()
object_missing_since = {}
object_added_since   = {}

active_warning = ""
warning_time   = 0

# ─── Person tracking ──────────────────────────────
# currently_present: { pid -> { name, last_seen_time, box } }
currently_present = {}

# pid -> time when they were LAST seen in frame
# used to confirm departure after FACE_ABSENT_SECS
person_last_seen   = {}   # pid -> timestamp

# pid -> stable display color (BGR)
_pid_colors = {}
_color_pool = [
    (0, 255, 200), (255, 200, 0), (200, 0, 255),
    (0, 200, 255), (255, 100, 0), (100, 255, 0),
    (0, 100, 255), (255, 0, 150), (0, 255, 100),
]

def pid_color(pid):
    if pid not in _pid_colors:
        idx = len(_pid_colors) % len(_color_pool)
        _pid_colors[pid] = _color_pool[idx]
    return _pid_colors[pid]

# =====================================================
# HELPERS
# =====================================================

def transparent_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

# =====================================================
# FACE PROCESSING FOR ONE PERSON BOX
# Returns (pid, display_name)
# =====================================================

def process_person_face(frame, x1, y1, x2, y2):
    """Crop the person box, run face recognition, return stable pid+name."""
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    try:
        rgb       = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return None, None

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return None, None

        pid, name, is_new = match_face(encodings[0])
        return pid, name

    except Exception as e:
        print("FACE ERROR:", e)
        return None, None

# =====================================================
# HANDLE PERSON ENTER / RETURN / UPDATE
# =====================================================

def handle_person_seen(pid, name, frame, x1, y1, x2, y2):
    """Called every detection frame a pid is visible."""
    now     = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    person_last_seen[pid] = now

    # Update DB
    faces_db[pid]["last_seen"]   = now_str
    faces_db[pid]["visit_count"] = faces_db[pid].get("visit_count", 0) + 1
    # Don't save every frame — save periodically (handled after loop)

    if pid not in currently_present:
        # New arrival or return
        was_seen_before = faces_db[pid].get("visit_count", 0) > 1
        currently_present[pid] = {"name": name, "box": (x1, y1, x2, y2)}

        if was_seen_before:
            event = "RETURNED"
            msg   = (f"🔄 PERSON RETURNED\n\n"
                     f"ID:    {pid}\n"
                     f"Name:  {name}\n"
                     f"Time:  {now_str}")
        else:
            event = "ENTERED"
            msg   = (f"🟢 NEW PERSON DETECTED\n\n"
                     f"ID:    {pid}\n"
                     f"Name:  {name}\n"
                     f"Time:  {now_str}")

        log_event(event, pid, name,
                  f"Visit #{faces_db[pid]['visit_count']}")
        async_alert(msg)

        # Save face capture
        cap_path = f"faces/captured/{pid}_{int(now)}.jpg"
        cv2.imwrite(cap_path, frame[y1:y2, x1:x2])
        if not faces_db[pid].get("known"):
            faces_db[pid]["photo"] = cap_path

        save_faces_db(faces_db)

    else:
        # Just update box position
        currently_present[pid]["box"] = (x1, y1, x2, y2)

# =====================================================
# HANDLE PERSON DEPARTURE (called in main loop)
# =====================================================

def check_departures(frame):
    now = time.time()
    departed = []

    for pid in list(currently_present.keys()):
        last = person_last_seen.get(pid, 0)
        if now - last > FACE_ABSENT_SECS:
            departed.append(pid)

    for pid in departed:
        data = currently_present.pop(pid)
        name = data["name"]
        cv2.imwrite(AFTER_IMG, frame)
        log_event("LEFT", pid, name,
                  f"Last seen {datetime.now().strftime('%H:%M:%S')}")
        async_alert(
            f"⚠️ PERSON LEFT\n\n"
            f"ID:    {pid}\n"
            f"Name:  {name}\n"
            f"Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        threading.Thread(target=play_alarm, daemon=True).start()

# =====================================================
# WINDOW
# =====================================================

cv2.namedWindow("AI SURVEILLANCE", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI SURVEILLANCE",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

log_event("SYSTEM START", "—", "—", "Surveillance system initialized")

# =====================================================
# MAIN LOOP
# =====================================================

_last_db_save = time.time()

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

    # ── Baseline ──────────────────────────────────
    if not baseline_saved and time.time() - startup_time > 5:
        cv2.imwrite(BEFORE_IMG, frame)
        baseline_saved = True
        confirmed_counts = last_seen_counter.copy()
        async_alert("✅ SURVEILLANCE ACTIVE\nBaseline saved. Monitoring started.")
        log_event("BASELINE", "—", "—", "Baseline snapshot saved")

    # ─────────────────────────────────────────────
    # YOLO DETECTION
    # ─────────────────────────────────────────────

    if frame_count % PROCESS_EVERY_N == 0:

        small   = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
        results = model.track(source=small, conf=CONFIDENCE,
                               persist=True, verbose=False)
        boxes   = results[0].boxes

        latest_detections = []
        current_objects   = []
        pids_seen_this_frame = set()

        sx = FRAME_WIDTH  / DETECTION_WIDTH
        sy = FRAME_HEIGHT / DETECTION_HEIGHT

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

            if label == "person":
                pid, name = process_person_face(frame, x1, y1, x2, y2)
                if pid:
                    pids_seen_this_frame.add(pid)
                    handle_person_seen(pid, name, frame, x1, y1, x2, y2)
                    disp_name = f"{pid} | {name}"

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

        # ── Departure check ───────────────────────
        check_departures(frame)

        # ── Non-person object tracking ─────────────
        if baseline_saved:
            now = time.time()

            for obj, conf_count in list(confirmed_counts.items()):
                cur_count = current_counter.get(obj, 0)
                if cur_count < conf_count:
                    if obj not in object_missing_since:
                        object_missing_since[obj] = now
                    if now - object_missing_since[obj] >= CONFIRMATION_TIME:
                        removed_n = conf_count - cur_count
                        confirmed_counts[obj] = cur_count
                        if cur_count == 0:
                            del confirmed_counts[obj]
                        del object_missing_since[obj]
                        lbl = f"{obj} x{removed_n}" if removed_n > 1 else obj
                        log_event("OBJ REMOVED", "—", lbl,
                                  f"{conf_count} → {cur_count}")
                        cv2.imwrite(AFTER_IMG, frame)
                        threading.Thread(target=play_alarm, daemon=True).start()
                        async_alert(
                            f"⚠️ OBJECT REMOVED\n\nObject: {lbl}\n"
                            f"Was: {conf_count} → Now: {cur_count}\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}"
                        )
                        active_warning = f"REMOVED: {lbl.upper()}"
                        warning_time   = now
                else:
                    object_missing_since.pop(obj, None)

            for obj, cur_count in current_counter.items():
                conf_count = confirmed_counts.get(obj, 0)
                if cur_count > conf_count:
                    if obj not in object_added_since:
                        object_added_since[obj] = now
                    if now - object_added_since[obj] >= CONFIRMATION_TIME:
                        added_n = cur_count - conf_count
                        confirmed_counts[obj] = cur_count
                        del object_added_since[obj]
                        lbl = f"{obj} x{cur_count}" if added_n > 1 or conf_count > 0 else obj
                        log_event("OBJ ADDED", "—", lbl,
                                  f"{conf_count} → {cur_count}")
                        async_alert(
                            f"🟢 OBJECT ADDED\n\nObject: {lbl}\n"
                            f"Was: {conf_count} → Now: {cur_count}\n"
                            f"Time: {datetime.now().strftime('%H:%M:%S')}"
                        )
                        active_warning = f"ADDED: {lbl.upper()}"
                        warning_time   = now
                else:
                    object_added_since.pop(obj, None)

    # ─────────────────────────────────────────────
    # DRAW BOUNDING BOXES
    # ─────────────────────────────────────────────

    if show_boxes:
        for d in latest_detections:
            x1, y1, x2, y2 = d["box"]
            pid   = d.get("pid")
            color = pid_color(pid) if pid else CYAN

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

            label_text = d["disp_name"]
            conf_text  = f"{d['conf']:.2f}"
            full_text  = f"{label_text}  {conf_text}"

            text_w = len(full_text) * 10
            cv2.rectangle(display,
                          (x1, y1 - 32), (x1 + text_w, y1),
                          color, -1)
            cv2.putText(display, full_text,
                        (x1 + 6, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 2)

    height, width = display.shape[:2]

    # ─────────────────────────────────────────────
    # TOP BAR
    # ─────────────────────────────────────────────

    transparent_rect(display, 0, 0, width, 58, BLACK, 0.88)

    cv2.putText(display, "AI SMART SURVEILLANCE",
                (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2)

    cv2.putText(display,
                f"Objects: {len(latest_detections)}  |  "
                f"Persons: {len(currently_present)}  |  "
                f"DB: {len(faces_db)} faces",
                (360, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)

    status_txt   = "READY" if baseline_saved else "SCANNING..."
    status_color = GREEN   if baseline_saved else RED
    cv2.putText(display, status_txt,
                (width - 140, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # ─────────────────────────────────────────────
    # WARNING BANNER
    # ─────────────────────────────────────────────

    if time.time() - warning_time < 3:
        bc = RED if "REMOVED" in active_warning else GREEN
        cv2.rectangle(display,
                      (width // 2 - 290, 62),
                      (width // 2 + 290, 112), bc, -1)
        cv2.putText(display, active_warning,
                    (width // 2 - 270, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 3)

    # ─────────────────────────────────────────────
    # LEFT PANEL — Live Objects + Present Persons
    # ─────────────────────────────────────────────

    if show_left_panel:
        transparent_rect(display, 0, 58, 230, height, DARK, 0.78)

        cv2.putText(display, "LIVE OBJECTS",
                    (12, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.72, CYAN, 2)
        y = 130
        for obj, count in object_counts.items():
            cv2.putText(display, f"  {obj}: {count}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
            y += 36

        y += 10
        cv2.putText(display, "PRESENT PERSONS",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, ORANGE, 2)
        y += 30
        for pid, data in currently_present.items():
            color = pid_color(pid)
            cv2.putText(display, f"  {pid}: {data['name'][:14]}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y += 28

    # ─────────────────────────────────────────────
    # RIGHT PANEL — Event Log
    # ─────────────────────────────────────────────

    if show_right_panel:
        transparent_rect(display, width - 360, 58, width, height, DARK, 0.78)

        cv2.putText(display, "EVENTS",
                    (width - 340, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, CYAN, 2)
        y = 130
        for event in reversed(history_log[-18:]):
            if "REMOVED" in event or "LEFT" in event:
                ec = RED
            elif "ADDED" in event or "ENTERED" in event or "RETURNED" in event:
                ec = GREEN
            else:
                ec = WHITE
            cv2.putText(display, event[:54],
                        (width - 350, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, ec, 1)
            y += 26

    # ─────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────

    if show_footer:
        transparent_rect(display, 0, height - 44, width, height, BLACK, 0.88)
        cv2.putText(display,
                    "L:Left Panel  R:Events  D:Boxes  B:Footer  S:Evidence  Q:Quit",
                    (18, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, WHITE, 1)

    # ─────────────────────────────────────────────
    # SHOW
    # ─────────────────────────────────────────────

    cv2.imshow("AI SURVEILLANCE", display)

    # ─────────────────────────────────────────────
    # KEY HANDLERS
    # ─────────────────────────────────────────────

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('l'):
        show_left_panel  = not show_left_panel
    elif key == ord('r'):
        show_right_panel = not show_right_panel
    elif key == ord('d'):
        show_boxes       = not show_boxes
    elif key == ord('b'):
        show_footer      = not show_footer
    elif key == ord('s'):
        log_event("EVIDENCE", "—", "—", "Manual evidence snapshot sent")
        async_alert("📸 SENDING EVIDENCE")
        if os.path.exists(BEFORE_IMG):
            async_photo(BEFORE_IMG, "📷 BEFORE")
        if os.path.exists(AFTER_IMG):
            async_photo(AFTER_IMG, "📷 AFTER")

# =====================================================
# CLEANUP
# =====================================================

save_faces_db(faces_db)
log_event("SYSTEM STOP", "—", "—", "Surveillance session ended")

# Write closing divider to event log
with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
    f.write(_table_divider() + "\n\n")
    f.write(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

cap.release()
cv2.destroyAllWindows()
print("System Closed")
