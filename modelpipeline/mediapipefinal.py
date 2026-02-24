import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pythonosc import udp_client
import time

# ─────────────────────────────
# CONFIG
# ─────────────────────────────
VMC_IP   = "127.0.0.1"
VMC_PORT = 39539
MODEL_PATH = "face_landmarker.task"

SQUINT_OFFSET = 0.2
BLINK_BOOST   = 1.4
BLINK_TRIGGER = 0.2

vmc_client = udp_client.SimpleUDPClient(VMC_IP, VMC_PORT)

# ─────────────────────────────
# CALLBACK
# ─────────────────────────────
def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):

    if not result.face_blendshapes:
        return

    for blendshape in result.face_blendshapes[0]:
        name  = blendshape.category_name
        score = float(blendshape.score)

        # Koreksi Squint
        if name in ["eyeSquintLeft", "eyeSquintRight"]:
            score = max(0.0, score - SQUINT_OFFSET)

        # Boost Blink
        if name in ["eyeBlinkLeft", "eyeBlinkRight"]:
            if score > BLINK_TRIGGER:
                score = min(1.0, score * BLINK_BOOST)

        vmc_client.send_message("/VMC/Ext/Blend/Val", [name, score])

# ─────────────────────────────
# MAIN
# ─────────────────────────────
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_face_blendshapes=True
)

cap = cv2.VideoCapture(3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # lebih ringan
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.setUseOptimized(True)
cap.set(cv2.CAP_PROP_FPS, 24)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)

cap.release()