import cv2
import sys

print("=" * 60)
print("DIAGNOSTIC TOOL - OBS VIRTUAL CAMERA")
print("=" * 60)

# Test semua backend yang tersedia
backends = [
    ("CAP_ANY (Default)", cv2.CAP_ANY),
    ("CAP_DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("CAP_MSMF (Media Foundation)", cv2.CAP_MSMF),
]

print("\n[1] Testing Camera Index 3 dengan berbagai backend:\n")

working_backend = None
for backend_name, backend_id in backends:
    print(f"Testing {backend_name}...", end=" ")
    cap = cv2.VideoCapture(3, backend_id)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ BERHASIL")
            print(f"   - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            print(f"   - Channels: {frame.shape[2] if len(frame.shape) > 2 else 'N/A'}")
            print(f"   - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            print(f"   - Backend: {cap.get(cv2.CAP_PROP_BACKEND)}")
            print(f"   - Format: {cap.get(cv2.CAP_PROP_FORMAT)}")
            print(f"   - Codec: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
            
            working_backend = (backend_name, backend_id)
            cap.release()
            break
        else:
            print(f"✗ GAGAL (opened tapi tidak bisa read frame)")
            cap.release()
    else:
        print(f"✗ GAGAL (tidak bisa open)")

print("\n" + "=" * 60)

if working_backend:
    print(f"\n[2] Backend yang WORK: {working_backend[0]}\n")
    
    # Test lebih detail dengan backend yang work
    cap = cv2.VideoCapture(3, working_backend[1])
    
    print("Properti kamera yang bisa diset:")
    properties_to_test = [
        ("FRAME_WIDTH", cv2.CAP_PROP_FRAME_WIDTH, 640),
        ("FRAME_HEIGHT", cv2.CAP_PROP_FRAME_HEIGHT, 480),
        ("FPS", cv2.CAP_PROP_FPS, 30),
        ("BUFFERSIZE", cv2.CAP_PROP_BUFFERSIZE, 1),
    ]
    
    for prop_name, prop_id, test_value in properties_to_test:
        before = cap.get(prop_id)
        cap.set(prop_id, test_value)
        after = cap.get(prop_id)
        
        if abs(after - test_value) < 0.1:
            print(f"   ✓ {prop_name}: {before} -> {after} (SET BERHASIL)")
        else:
            print(f"   ✗ {prop_name}: {before} -> {after} (SET GAGAL, target: {test_value})")
    
    print("\n[3] Testing continuous frame read (5 frames):\n")
    
    for i in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"   Frame {i+1}: ✓ {frame.shape}")
        else:
            print(f"   Frame {i+1}: ✗ FAILED")
    
    cap.release()
    
    print("\n" + "=" * 60)
    print("KESIMPULAN:")
    print(f"✓ Gunakan backend: {working_backend[0]}")
    print(f"✓ Backend ID untuk kode: cv2.VideoCapture(3, {working_backend[1]})")
    print("=" * 60)
    
else:
    print("\n[ERROR] TIDAK ADA BACKEND YANG BERHASIL!\n")
    print("Troubleshooting:")
    print("1. Pastikan OBS Virtual Camera sudah di-START")
    print("2. Coba restart OBS")
    print("3. Coba index kamera lain (0, 1, 2, 4)")
    print("4. Check apakah aplikasi lain bisa akses OBS Virtual Camera")
    print("\n[4] Scanning semua available camera indexes:\n")
    
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            status = "✓ CAN READ" if ret else "✗ CAN'T READ"
            print(f"   Index {idx}: {status}")
            cap.release()
        else:
            print(f"   Index {idx}: NOT AVAILABLE")
    
    print("\n" + "=" * 60)