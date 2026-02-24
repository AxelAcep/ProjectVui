import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pythonosc import udp_client
import time
import json
import csv
import os
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
vmc_client = udp_client.SimpleUDPClient("127.0.0.1", 39539)
model_path = 'face_landmarker.task'

# ─────────────────────────────────────────────
# FACE REGION — Landmark index per bagian wajah
# Berdasarkan MediaPipe Face Mesh 478 landmarks
# ─────────────────────────────────────────────
FACE_REGIONS = {
    "LEFT_EYE":   [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "RIGHT_EYE":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "LEFT_BROW":  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "RIGHT_BROW": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "NOSE":       [1, 2, 5, 4, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18],
    "MOUTH":      [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409],
    "LEFT_CHEEK": [116, 123, 147, 213, 192, 214, 210, 211, 32],
    "RIGHT_CHEEK":[345, 352, 376, 433, 416, 434, 430, 431, 262],
    "CHIN":       [152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54],
    "FOREHEAD":   [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
}

# ─────────────────────────────────────────────
# BLENDSHAPE GROUPING
# ─────────────────────────────────────────────
BLENDSHAPE_GROUPS = {
    "EYE": [
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight",
        "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight",
        "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight",
        "eyeWideLeft", "eyeWideRight",
    ],
    "BROW": [
        "browDownLeft", "browDownRight",
        "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight",
    ],
    "MOUTH": [
        "jawForward", "jawLeft", "jawRight", "jawOpen",
        "mouthClose", "mouthFunnel", "mouthPucker",
        "mouthLeft", "mouthRight",
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "mouthDimpleLeft", "mouthDimpleRight",
        "mouthStretchLeft", "mouthStretchRight",
        "mouthRollLower", "mouthRollUpper",
        "mouthShrugLower", "mouthShrugUpper",
        "mouthPressLeft", "mouthPressRight",
        "mouthLowerDownLeft", "mouthLowerDownRight",
        "mouthUpperUpLeft", "mouthUpperUpRight",
    ],
    "CHEEK": [
        "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    ],
    "NOSE": [
        "noseSneerLeft", "noseSneerRight",
    ],
    "HEAD": [
        "headRoll", "headPitch", "headYaw",  # kalau ada
    ],
    "TONGUE": [
        "tongueOut",
    ],
}

# Buat reverse lookup: name → group
BLENDSHAPE_TO_GROUP = {}
for group, names in BLENDSHAPE_GROUPS.items():
    for name in names:
        BLENDSHAPE_TO_GROUP[name] = group

# ─────────────────────────────────────────────
# CHEEKPUFF CALIBRATION
# mouthPucker dijadikan proxy karena cheekPuff
# tidak terdeteksi oleh MediaPipe di wajah ini.
#
# Cara kerja:
#   - Kalau mouthPucker >= CHEEK_THRESHOLD → trigger cheekPuff
#   - Nilai cheekPuff yang dikirim = proporsional dari mouthPucker
#     dihitung dari range CHEEK_THRESHOLD sampai 1.0
#   - Ubah CHEEK_THRESHOLD sesuai kebutuhanmu
# ─────────────────────────────────────────────
CHEEK_THRESHOLD_ON  = 0.72  # mouthPucker untuk NYALAKAN cheekPuff
CHEEK_THRESHOLD_OFF = 0.60  # mouthPucker untuk MATIKAN cheekPuff (hysteresis)
CHEEK_MAX           = 1.0   # nilai mouthPucker yang dianggap "penuh"
CHEEK_OUT_MAX       = 1.0   # nilai cheekPuff output maksimum ke VSeeFace
CHEEK_SMOOTH_FRAMES = 6     # jumlah frame untuk smoothing (lebih besar = lebih halus)

# State internal cheekPuff
_cheek_active       = False         # apakah sedang dalam kondisi "on"
_cheek_history      = []            # buffer smoothing
_cheek_last_value   = 0.0          # nilai terakhir yang dikirim

def compute_cheek_puff(mouth_pucker: float) -> float:
    """
    Hitung nilai cheekPuff dengan hysteresis + smoothing.

    Hysteresis:
      - Nyala  jika mouthPucker >= CHEEK_THRESHOLD_ON
      - Mati   jika mouthPucker <  CHEEK_THRESHOLD_OFF
      - Di antara keduanya → ikut state sebelumnya (tidak flickering)

    Smoothing:
      - Rata-rata dari CHEEK_SMOOTH_FRAMES terakhir
      - Gerakan naik/turun jadi lebih gradual
    """
    global _cheek_active, _cheek_history, _cheek_last_value

    # 1. Hysteresis — tentukan apakah aktif atau tidak
    if mouth_pucker >= CHEEK_THRESHOLD_ON:
        _cheek_active = True
    elif mouth_pucker < CHEEK_THRESHOLD_OFF:
        _cheek_active = False
    # zona tengah (0.60–0.72): ikut state sebelumnya, tidak berubah

    # 2. Hitung nilai raw proporsional
    if _cheek_active:
        ratio = (mouth_pucker - CHEEK_THRESHOLD_ON) / (CHEEK_MAX - CHEEK_THRESHOLD_ON)
        raw = min(max(ratio * CHEEK_OUT_MAX, 0.0), CHEEK_OUT_MAX)
    else:
        raw = 0.0

    # 3. Smoothing — simpan ke buffer dan ambil rata-rata
    _cheek_history.append(raw)
    if len(_cheek_history) > CHEEK_SMOOTH_FRAMES:
        _cheek_history.pop(0)

    smoothed = sum(_cheek_history) / len(_cheek_history)
    _cheek_last_value = round(smoothed, 4)
    return _cheek_last_value

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
latest_landmarks = None
latest_blendshapes = {}   # { name: score }
is_recording = False
record_session = []       # list of snapshot dicts
session_start_time = None

# ─────────────────────────────────────────────
# CALLBACK
# ─────────────────────────────────────────────
def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks, latest_blendshapes

    if result.face_landmarks:
        latest_landmarks = result.face_landmarks

    if result.face_blendshapes:
        blendshapes_this_frame = {}
        for blendshape in result.face_blendshapes[0]:
            name = blendshape.category_name
            score = float(blendshape.score)
            blendshapes_this_frame[name] = score
            vmc_client.send_message("/VMC/Ext/Blend/Val", [name, score])

        # ── CHEEKPUFF OVERRIDE ──────────────────────────────
        # MediaPipe tidak mendeteksi cheekPuff di wajah ini,
        # jadi kita hitung manual dari mouthPucker sebagai proxy.
        mouth_pucker = blendshapes_this_frame.get("mouthPucker", 0.0)
        cheek_value  = compute_cheek_puff(mouth_pucker)
        

        # Simpan ke dict supaya muncul di HUD & snapshot
        blendshapes_this_frame["cheekPuff"] = cheek_value

        # Kirim ke VSeeFace (override nilai default yang 0)
        vmc_client.send_message("/VMC/Ext/Blend/Val", ["cheekPuff", cheek_value])
        # ───────────────────────────────────────────────────

        latest_blendshapes = blendshapes_this_frame

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_region_center(landmarks, region_indices, img_w, img_h):
    """Hitung titik tengah dari sekumpulan landmark."""
    xs, ys = [], []
    for idx in region_indices:
        if idx < len(landmarks):
            lm = landmarks[idx]
            xs.append(lm.x * img_w)
            ys.append(lm.y * img_h)
    if xs:
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
    return None

def draw_region_labels(frame, landmarks, img_w, img_h):
    """Gambar label nama region di atas wajah."""
    for region_name, indices in FACE_REGIONS.items():
        center = get_region_center(landmarks, indices, img_w, img_h)
        if center:
            cx, cy = center
            # Background label
            label = region_name.replace("_", " ")
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(frame, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2), (20, 20, 20), -1)
            cv2.putText(frame, label, (cx, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)

def draw_blendshape_hud(frame, blendshapes, img_h):
    """Tampilkan nilai blendshape per grup di sudut kiri atas."""
    x_start = 10
    y = 20
    line_h = 16
    col_width = 240

    grouped = {}
    for name, score in blendshapes.items():
        group = BLENDSHAPE_TO_GROUP.get(name, "OTHER")
        if group not in grouped:
            grouped[group] = []
        grouped[group].append((name, score))

    col = 0
    for group_name, items in grouped.items():
        x = x_start + col * col_width
        # Header grup
        cv2.putText(frame, f"[ {group_name} ]", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 50), 1, cv2.LINE_AA)
        local_y = y + line_h
        for name, score in sorted(items, key=lambda i: -i[1]):
            if score > 0.01:  # Hanya tampilkan yang aktif
                bar_len = int(score * 80)
                # Warna khusus untuk cheekPuff override
                if name == "cheekPuff":
                    color = (255, 100, 0)  # oranye = hasil kalkulasi manual
                else:
                    color = (0, 200, 100) if score < 0.5 else (0, 100, 255) if score < 0.8 else (0, 50, 255)
                cv2.rectangle(frame, (x, local_y - 9), (x + bar_len, local_y - 2), color, -1)
                short_name = name.replace("Left","L").replace("Right","R").replace("mouth","m").replace("eye","e").replace("brow","br")
                suffix = " [proxy]" if name == "cheekPuff" else ""
                cv2.putText(frame, f"{short_name}{suffix}: {score:.2f}", (x + bar_len + 3, local_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
                local_y += line_h
                if local_y > img_h - 20:
                    break
        col += 1
        if col > 3:
            col = 0
            y += 200

def take_snapshot(landmarks_list, blendshapes):
    """Buat satu snapshot data untuk dicatat."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "blendshapes": {},
        "landmark_regions": {}
    }

    # Blendshapes
    for name, score in blendshapes.items():
        group = BLENDSHAPE_TO_GROUP.get(name, "OTHER")
        if group not in snapshot["blendshapes"]:
            snapshot["blendshapes"][group] = {}
        snapshot["blendshapes"][group][name] = round(score, 4)

    # Koordinat landmark per region
    if landmarks_list:
        for face_landmarks in landmarks_list:
            for region_name, indices in FACE_REGIONS.items():
                coords = []
                for idx in indices:
                    if idx < len(face_landmarks):
                        lm = face_landmarks[idx]
                        coords.append({"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)})
                snapshot["landmark_regions"][region_name] = coords

    return snapshot

def print_snapshot_to_terminal(snapshot):
    """Print snapshot ke terminal dengan format rapi."""
    print("\n" + "="*60)
    print(f"📸 SNAPSHOT — {snapshot['timestamp']}")
    print("="*60)

    print("\n▶ BLENDSHAPES (hanya yang aktif > 0.05):")
    for group, items in snapshot["blendshapes"].items():
        active = {k: v for k, v in items.items() if v > 0.05}
        if active:
            print(f"  [{group}]")
            for name, val in sorted(active.items(), key=lambda x: -x[1]):
                bar = "█" * int(val * 20)
                print(f"    {name:<35} {val:.3f}  {bar}")

    print("\n▶ LANDMARK REGION CENTERS (normalized 0.0–1.0):")
    for region, coords in snapshot["landmark_regions"].items():
        if coords:
            avg_x = round(sum(c["x"] for c in coords) / len(coords), 3)
            avg_y = round(sum(c["y"] for c in coords) / len(coords), 3)
            print(f"  {region:<15} center = (x={avg_x}, y={avg_y})")
    print("="*60)

def save_session_to_file(session_data):
    """Simpan seluruh sesi ke JSON dan CSV."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("recordings", exist_ok=True)

    # JSON — lengkap
    json_path = f"recordings/session_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(session_data, f, indent=2)
    print(f"\n💾 JSON saved → {json_path}")

    # CSV — blendshape flat per snapshot
    csv_path = f"recordings/session_{ts}_blendshapes.csv"
    all_names = set()
    for snap in session_data:
        for group_items in snap["blendshapes"].values():
            all_names.update(group_items.keys())
    all_names = sorted(all_names)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + all_names)
        for snap in session_data:
            flat = {}
            for group_items in snap["blendshapes"].values():
                flat.update(group_items)
            row = [snap["timestamp"]] + [flat.get(n, 0.0) for n in all_names]
            writer.writerow(row)
    print(f"💾 CSV  saved → {csv_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_face_blendshapes=True
)

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 24)

print("╔══════════════════════════════════════════════════╗")
print("║       VTuber Face Tracker + Recorder            ║")
print("╠══════════════════════════════════════════════════╣")
print("║  R  → Toggle Record ON/OFF                      ║")
print("║  S  → Snapshot (saat recording)                 ║")
print("║  Q  → Quit & save semua data                    ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  cheekPuff proxy via mouthPucker                ║")
print(f"║  Nyala  : mouthPucker >= {CHEEK_THRESHOLD_ON}                  ║")
print(f"║  Mati   : mouthPucker <  {CHEEK_THRESHOLD_OFF}                  ║")
print(f"║  Smooth : {CHEEK_SMOOTH_FRAMES} frames                              ║")
print("╚══════════════════════════════════════════════════╝\n")

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Draw landmarks mesh
        if latest_landmarks:
            for face_landmarks in latest_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z)
                    for l in face_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks_proto,
                    connections=solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
            # Label region
            draw_region_labels(frame, latest_landmarks[0], img_w, img_h)

        # HUD blendshape
        if latest_blendshapes:
            draw_blendshape_hud(frame, latest_blendshapes, img_h)

        # Status recording
        rec_color = (0, 0, 255) if is_recording else (100, 100, 100)
        rec_text = f"● REC [{len(record_session)} snapshots]" if is_recording else "○ IDLE  (R=record, S=snapshot, Q=quit)"
        cv2.putText(frame, rec_text, (img_w - 380, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 2, cv2.LINE_AA)

        cv2.imshow('VuiTuber Pipeline', frame)

        key = cv2.waitKey(1) & 0xFF

        # R → toggle recording
        if key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                session_start_time = datetime.now()
                print(f"\n🔴 Recording DIMULAI — {session_start_time.strftime('%H:%M:%S')}")
                print("Tekan S untuk snapshot, R lagi untuk stop.\n")
            else:
                print(f"\n⏹  Recording DIHENTIKAN — {len(record_session)} snapshots tersimpan.")

        # S → snapshot manual
        elif key == ord('s'):
            if is_recording and latest_blendshapes and latest_landmarks:
                snap = take_snapshot(latest_landmarks, latest_blendshapes)
                record_session.append(snap)
                print_snapshot_to_terminal(snap)
                print(f"[Total snapshots: {len(record_session)}]")
            elif not is_recording:
                print("⚠ Aktifkan recording dulu dengan tekan R!")

        # Q → quit + save
        elif key == ord('q'):
            break

    # Simpan file setelah keluar
    if record_session:
        print(f"\n📦 Menyimpan {len(record_session)} snapshots...")
        save_session_to_file(record_session)
    else:
        print("\nTidak ada data yang direcord.")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Pipeline selesai.")