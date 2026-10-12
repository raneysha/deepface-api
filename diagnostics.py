"""Startup diagnostics: lightweight, imported first so the numbers are printed before TensorFlow loads.
Helps tell apart 'crashed with an error' from 'killed for using too much memory' in Railway's logs."""
import json
import os
import sys
import threading
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[startup] {msg}", flush=True)


def _read_mb(path: str):
    try:
        v = open(path).read().strip()
    except OSError:
        return None
    return int(v) // (1024 * 1024) if v.isdigit() and int(v) < (1 << 60) else None  # "max" / huge = no limit


def memory_limit_mb():
    """Container memory limit (cgroup v2, then v1), or None if there is none."""
    return _read_mb("/sys/fs/cgroup/memory.max") or _read_mb("/sys/fs/cgroup/memory/memory.limit_in_bytes")


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def peak_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def memory_summary() -> str:
    lim = memory_limit_mb()
    return (f"memory in use: {rss_mb() or '?'} MB (peak {peak_mb() or '?'} MB), "
            + (f"container limit: {lim} MB" if lim else "no container limit detected"))


# ---- crash trail -------------------------------------------------------------------------------------------------
# When a container is killed for using too much memory it dies instantly and prints nothing. So every face job leaves a small
# note on the persistent volume while it runs and removes it when it finishes. If a note is still there at the next start, the
# previous run ended in the middle of that job, and we say so in the log.
_STATE = Path(os.getenv("DEEPFACE_HOME", os.path.expanduser("~"))) / ".deepface" / "inflight.json"
_jobs: dict = {}
_lock = threading.Lock()


def _persist() -> None:
    try:
        if _jobs:
            _STATE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(_jobs))
            os.replace(tmp, _STATE)
        else:
            _STATE.unlink(missing_ok=True)
    except OSError:
        pass   # diagnostics must never break a request


def job_started(kind: str, detail: str = "") -> str:
    jid = f"{time.time_ns()}-{threading.get_ident()}"
    with _lock:
        _jobs[jid] = {"kind": kind, "detail": detail, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "rss_mb": rss_mb(), "limit_mb": memory_limit_mb()}
        _persist()
    return jid


def job_finished(jid: str) -> None:
    with _lock:
        _jobs.pop(jid, None)
        _persist()


def report_previous_run() -> None:
    """Call once at startup, before any job runs."""
    try:
        data = json.loads(_STATE.read_text())
    except (OSError, ValueError):
        return
    try:
        _STATE.unlink(missing_ok=True)
    except OSError:
        pass
    for j in data.values():
        log(f"PREVIOUS RUN ENDED MID-JOB: {j.get('kind')} ({j.get('detail')}) started {j.get('started')} when memory was "
            f"{j.get('rss_mb')} MB of {j.get('limit_mb') or 'no detected'} MB. It was most likely killed for running out of memory "
            "(or this was just a redeploy that interrupted it).")


def start_memory_monitor(interval: float = 2.0) -> None:
    """Warn in the log while memory climbs towards the container limit, before the kill that leaves no message."""
    limit = memory_limit_mb()
    if not limit:
        return

    def loop():
        last = 0.0
        while True:
            time.sleep(interval)
            used = rss_mb()
            if used and used > 0.85 * limit and time.time() - last > 10:
                last = time.time()
                log(f"WARNING: memory {used} MB of {limit} MB limit; the container is killed and restarted if it reaches the limit. "
                    f"Jobs running: {len(_jobs)}")

    threading.Thread(target=loop, daemon=True, name="memory-monitor").start()


log(f"python {sys.version.split()[0]}; {memory_summary()}")
def on_volume(path: str):
    """True if `path` (or its nearest existing parent) lives on a different filesystem than the container's own: a mounted volume."""
    p = os.path.abspath(path)
    while not os.path.exists(p) and p != "/":
        p = os.path.dirname(p)
    try:
        return os.stat(p).st_dev != os.stat("/").st_dev
    except OSError:
        return None


def storage_summary() -> str:
    parts = []
    for name, default in (("UPLOAD_DIR", "storage/uploads"), ("DEEPFACE_HOME", os.path.expanduser("~"))):
        path = os.getenv(name, default)
        vol = on_volume(path)
        parts.append(f"{name}={os.path.abspath(path)} (" + ("on a mounted volume: files survive redeploys" if vol else
                     "NOT on a mounted volume: files are lost on every redeploy" if vol is False else "unknown") + ")")
    return "; ".join(parts)


log(f"settings present: JWT_SECRET={'yes (%d chars)' % len(os.getenv('JWT_SECRET', '')) if os.getenv('JWT_SECRET') else 'NO'}, "
    f"DEEPFACE_POSTGRES_URI={'yes' if os.getenv('DEEPFACE_POSTGRES_URI') else 'NO'}, PORT={os.getenv('PORT', 'not set')}")

log("storage: " + storage_summary())
