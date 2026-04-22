"""
student/gpu_keepalive.py  — tight loop, no sleep
"""
import threading
import time
import torch


def _keepalive_loop():
    if not torch.cuda.is_available():
        print("[keepalive] No GPU found.", flush=True)
        while True:
            time.sleep(30)

    device = torch.device("cuda")
    A = torch.randn(4096, 4096, device=device)
    B = torch.randn(4096, 4096, device=device)
    print(f"[keepalive] Running on {torch.cuda.get_device_name(0)}", flush=True)

    i = 0
    while True:
        # No sleep — continuous matmuls keep SM utilization high
        A = torch.mm(A, B)
        i += 1
        if i % 500 == 0:
            print(f"[keepalive] alive (iter {i})", flush=True)


def start() -> threading.Thread:
    t = threading.Thread(target=_keepalive_loop, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    _keepalive_loop()
