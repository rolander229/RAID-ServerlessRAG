"""Diagnose the runtime worker handshake in isolation."""
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener

AUTH = b"m2s-runtime"
key = sys.argv[1] if len(sys.argv) > 1 else "recsys_deepfm"

d = tempfile.mkdtemp(prefix="m2s-diag-")
addr = os.path.join(d, "s.sock")
lis = Listener(addr, family="AF_UNIX", authkey=AUTH)

env = dict(os.environ)
env["PYTHONPATH"] = "M2S/src"
env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
logp = os.path.join(d, "w.log")
log = open(logp, "wb")
proc = subprocess.Popen([sys.executable, "-m", "workloads.worker", key, addr],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
print("worker pid:", proc.pid)

try:
    conn = lis.accept()
    print("handshake:", conn.recv())
    conn.send({"ops": ["parse_user_context"], "inbound": {}, "outbound": []})
    print("reply:", conn.recv())
except Exception as e:
    print("DRIVER ERROR:", type(e).__name__, str(e)[:200])

time.sleep(0.5)
print("worker exit code:", proc.poll())
log.close()
print("---- worker stdout/stderr ----")
print(open(logp).read()[-4000:])
