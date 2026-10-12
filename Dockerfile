FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

# OpenCV needs these shared libraries, which the slim image doesn't include.
# (The glib package was renamed in newer Debian releases, so try both names.)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 \
 && (apt-get install -y --no-install-recommends libglib2.0-0t64 || apt-get install -y --no-install-recommends libglib2.0-0) \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code-only changes rebuild in seconds. constraints.txt pins exact, tested versions.
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY . .

# /data is where the Railway volume is mounted: uploaded photos and DeepFace's downloaded models live there,
# so they survive redeploys. Override in the service variables if you mount the volume elsewhere.
ENV UPLOAD_DIR=/data/uploads \
    DEEPFACE_HOME=/data

# Railway sets PORT. One worker on purpose (login throttling and the face-job limit are per process).
# --forwarded-allow-ips="*": the app is only reachable through Railway's proxy, so trust its X-Forwarded-* headers
# (this makes login throttling see real client IPs).
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
