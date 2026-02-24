import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pythonosc import udp_client
import time

vmc_client = udp_client.SimpleUDPClient("127.0.0.1", 39539)
model_path = 'face_landmarker.task'
latest_landmarks = None

def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks
    # Landmarks akan tetap ada di 'result' meskipun tidak di-set di options
    if result.face_landmarks:
        latest_landmarks = result.face_landmarks
        
    if result.face_blendshapes:
        for blendshape in result.face_blendshapes[0]:
            vmc_client.send_message("/VMC/Ext/Blend/Val", [blendshape.category_name, float(blendshape.score)])

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_face_blendshapes=True
)

cap = cv2.VideoCapture(3) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Mencegah lag/freeze

cap.set(cv2.CAP_PROP_FPS, 24)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_landmarks:
            for face_landmarks in latest_landmarks:
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) for l in face_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks_proto,
                    connections=solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )

        cv2.imshow('VuiTuber Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()