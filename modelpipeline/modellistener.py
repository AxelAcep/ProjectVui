"""
VSeeFace OSC Listener
─────────────────────
Jalankan file ini di terminal TERPISAH, bersamaan dengan pipeline utama.

    python vseeface_listener.py

Akan menangkap semua pesan VMC yang VSeeFace kirim balik,
dan filter khusus blendshape yang aktif > threshold supaya
tidak kebanjiran output.
"""

from pythonosc import dispatcher, osc_server
import threading
import time
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LISTEN_PORT      = 39540   # port VSeeFace sender
ACTIVE_THRESHOLD = 0.05    # hanya tampilkan blendshape > nilai ini
CHEEK_FOCUS_MODE = True    # kalau True, highlight cheekPuff & mouthPucker

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
blendshape_state = {}   # { name: value } — snapshot terbaru
last_print_time  = 0
PRINT_INTERVAL   = 0.15  # detik antar print (supaya terminal tidak scroll gila)

# ─────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────
def handle_blendshape(address, *args):
    """Tangkap /VMC/Ext/Blend/Val"""
    global blendshape_state
    if len(args) >= 2:
        name  = str(args[0])
        value = float(args[1])
        blendshape_state[name] = value

def handle_blend_apply(address, *args):
    """
    /VMC/Ext/Blend/Apply dikirim VSeeFace setelah semua blendshape di-set.
    Ini momen terbaik untuk print snapshot karena semua nilai sudah lengkap.
    """
    global last_print_time
    now = time.time()
    if now - last_print_time < PRINT_INTERVAL:
        return
    last_print_time = now
    print_active_blendshapes()

def handle_any(address, *args):
    """Catch-all untuk lihat address lain yang dikirim VSeeFace."""
    # Uncomment baris di bawah kalau mau lihat SEMUA pesan (verbose banget)
    # print(f"[OTHER] {address} → {args}")
    pass

def print_active_blendshapes():
    """Print blendshape yang aktif dengan highlight khusus untuk cheek & mouth."""
    if not blendshape_state:
        return

    active = {k: v for k, v in blendshape_state.items() if v > ACTIVE_THRESHOLD}
    if not active:
        return

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n── {ts} ──────────────────────────────")

    # Sort by value descending
    for name, val in sorted(active.items(), key=lambda x: -x[1]):
        bar = "█" * int(val * 25)

        # Highlight khusus
        if CHEEK_FOCUS_MODE and name == "cheekPuff":
            prefix = "🟠 CHEEK  "
        elif CHEEK_FOCUS_MODE and name == "mouthPucker":
            prefix = "🔵 PUCKER "
        elif CHEEK_FOCUS_MODE and "mouth" in name.lower():
            prefix = "   mouth  "
        elif CHEEK_FOCUS_MODE and "eye" in name.lower():
            prefix = "   eye    "
        else:
            prefix = "          "

        print(f"  {prefix} {name:<35} {val:.3f}  {bar}")

# ─────────────────────────────────────────────
# SETUP DISPATCHER
# ─────────────────────────────────────────────
disp = dispatcher.Dispatcher()
disp.map("/VMC/Ext/Blend/Val",   handle_blendshape)
disp.map("/VMC/Ext/Blend/Apply", handle_blend_apply)
disp.set_default_handler(handle_any)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print("╔══════════════════════════════════════════════════╗")
print("║         VSeeFace OSC Listener                   ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  Listening port : {LISTEN_PORT}                          ║")
print(f"║  Show threshold : > {ACTIVE_THRESHOLD}                         ║")
print(f"║  Cheek focus    : {'ON ' if CHEEK_FOCUS_MODE else 'OFF'}                           ║")
print("╠══════════════════════════════════════════════════╣")
print("║  Ctrl+C untuk stop                              ║")
print("╚══════════════════════════════════════════════════╝")
print(f"\nMenunggu data dari VSeeFace di port {LISTEN_PORT}...\n")

server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", LISTEN_PORT), disp)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n\n✅ Listener dihentikan.")
    server.shutdown()