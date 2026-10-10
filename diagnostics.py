"""Startup diagnostics: lightweight, imported first so the numbers are printed before TensorFlow loads.
Helps tell apart 'crashed with an error' from 'killed for using too much memory' in Railway's logs."""
import os
import sys


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


def memory_summary() -> str:
    lim = memory_limit_mb()
    return f"memory in use: {rss_mb() or '?'} MB, " + (f"container limit: {lim} MB" if lim else "no container limit detected")


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