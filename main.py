"""
FastAPI wrapper around DeepFace's database-backed API:
register / identify / search / build_index.

Embeddings live in the database (postgres by default), not on local disk,
so the service itself is stateless and can be scaled horizontally.
"""
import inspect
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from linking import Photographer, Upload, attach_owners, get_db, init_db

# --------------------------------------------------------------------------- #
# Config (env vars)
# --------------------------------------------------------------------------- #
DEFAULT_DB = os.getenv("FACE_DB_TYPE", "postgres")        # postgres | mongo | pgvector | weaviate | neo4j ...
DEFAULT_MODEL = os.getenv("FACE_MODEL", "Facenet")
DEFAULT_DETECTOR = os.getenv("FACE_DETECTOR", "opencv")
API_KEY = os.getenv("API_KEY")                             # if unset, auth is disabled
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "storage/uploads"))   # original images + thumbnails

# The DB connection itself is read by deepface (see README / docker-compose.yml),
# e.g. DEEPFACE_POSTGRES_URI. Optionally pass it explicitly:
CONNECTION_DETAILS = os.getenv("FACE_DB_CONNECTION")

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()  # creates photographers / uploads tables if missing
    yield


app = FastAPI(title="DeepFace DB API", version="1.1.0", lifespan=lifespan)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: Optional[str] = Security(_api_key_header)) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def read_image(upload: UploadFile, with_bytes: bool = False):
    data = await upload.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"{upload.filename}: larger than {MAX_UPLOAD_MB} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, f"{upload.filename}: not a decodable image")
    return (img, data) if with_bytes else img  # BGR ndarray, which DeepFace accepts directly


_MAGIC = ((b"\xff\xd8\xff", ".jpg", "image/jpeg"), (b"\x89PNG\r\n\x1a\n", ".png", "image/png"), (b"BM", ".bmp", "image/bmp"))


def prepare_original(img: np.ndarray, data: bytes):
    """Keep the uploaded bytes as-is for common formats; otherwise re-encode to JPEG."""
    for magic, ext, mime in _MAGIC:
        if data.startswith(magic):
            return data, ext, mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, ".webp", "image/webp"
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes(), ".jpg", "image/jpeg"


def make_thumb(img: np.ndarray, max_side: int = 320) -> bytes:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def call_deepface(fn, **kwargs):
    """
    Call a DeepFace function, silently dropping kwargs it doesn't support.
    Keeps this service working across deepface versions (e.g. the ANN flag has
    been documented as both `search_method` and `search_type`).
    """
    params = inspect.signature(fn).parameters
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
    clean = {k: v for k, v in kwargs.items() if v is not None and (accepts_var_kw or k in params)}
    return fn(**clean)


def jsonable(obj: Any) -> Any:
    """Convert DataFrames / numpy types into plain JSON."""
    if hasattr(obj, "to_json"):  # pandas DataFrame
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


async def run(fn, **kwargs):
    """Run blocking DeepFace code off the event loop and map errors to HTTP."""
    try:
        return jsonable(await run_in_threadpool(call_deepface, fn, **kwargs))
    except ValueError as e:  # e.g. "Face could not be detected"
        raise HTTPException(422, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


def common(model_name, detector_backend, align, enforce_detection, database_type):
    return dict(
        model_name=model_name,
        detector_backend=detector_backend,
        align=align,
        enforce_detection=enforce_detection,
        database_type=database_type,
        connection_details=CONNECTION_DETAILS,
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "db": DEFAULT_DB, "model": DEFAULT_MODEL}


@app.post("/register", dependencies=[Depends(require_key)])
async def register(
    files: List[UploadFile] = File(..., description="One or more face images"),
    model_name: str = Form(DEFAULT_MODEL),
    detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True),
    enforce_detection: bool = Form(True),
    database_type: str = Form(DEFAULT_DB),
):
    """Detect faces, compute embeddings and store them in the database."""
    imgs = [await read_image(f) for f in files]
    result = await run(
        DeepFace.register,
        img=imgs if len(imgs) > 1 else imgs[0],
        **common(model_name, detector_backend, align, enforce_detection, database_type),
    )
    return {"registered_files": [f.filename for f in files], "result": result}


@app.post("/identify", dependencies=[Depends(require_key)])
async def identify(
    file: UploadFile = File(...),
    identity_id: str = Form(..., description="ID of a registered identity (1:1 check)"),
    model_name: str = Form(DEFAULT_MODEL),
    detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True),
    enforce_detection: bool = Form(True),
    database_type: str = Form(DEFAULT_DB),
):
    """1:1 - is this face the registered identity `identity_id`?"""
    if not hasattr(DeepFace, "identify"):
        raise HTTPException(501, "Installed deepface has no identify(); install from the GitHub master branch.")
    img = await read_image(file)
    return await run(
        DeepFace.identify,
        img=img,
        identity_id=identity_id,
        **common(model_name, detector_backend, align, enforce_detection, database_type),
    )


@app.post("/search", dependencies=[Depends(require_key)])
async def search(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL),
    detector_backend: str = Form(DEFAULT_DETECTOR),
    distance_metric: str = Form("cosine"),
    align: bool = Form(True),
    enforce_detection: bool = Form(True),
    ann: bool = Form(False, description="Approximate nearest neighbour (needs /index for postgres/mongo)"),
    enrich: bool = Form(True, description="Attach upload + photographer data to each match"),
    database_type: str = Form(DEFAULT_DB),
    db: Session = Depends(get_db),
):
    """1:N - find the closest registered faces."""
    img = await read_image(file)
    extra = {}
    if ann:
        extra = {"search_method": "ann", "search_type": "ann"}  # unsupported one is dropped
    matches = await run(
        DeepFace.search,
        img=img,
        distance_metric=distance_metric,
        **extra,
        **common(model_name, detector_backend, align, enforce_detection, database_type),
    )
    if enrich:
        attach_owners(matches, db)
    return {"matches": matches}  # one list of matches per face found in the image


@app.post("/index", dependencies=[Depends(require_key)])
async def build_index(
    model_name: str = Form(DEFAULT_MODEL),
    detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True),
    database_type: str = Form(DEFAULT_DB),
):
    """Build the ANN index (FAISS) - needed for postgres/mongo; re-run after adding identities."""
    await run(
        DeepFace.build_index,
        model_name=model_name,
        detector_backend=detector_backend,
        align=align,
        database_type=database_type,
        connection_details=CONNECTION_DETAILS,
    )
    return {"status": "index built"}


# --------------------------------------------------------------------------- #
# Photographers & uploads (relational data linked to face embeddings)
# --------------------------------------------------------------------------- #
class PhotographerIn(BaseModel):
    name: str
    email: Optional[str] = None


@app.post("/photographers", status_code=201, dependencies=[Depends(require_key)])
def create_photographer(body: PhotographerIn, db: Session = Depends(get_db)):
    if body.email and db.scalar(select(Photographer).where(Photographer.email == body.email)):
        raise HTTPException(409, "A photographer with this email already exists")
    p = Photographer(name=body.name, email=body.email)
    db.add(p)
    db.commit()
    return {"id": p.id, "name": p.name, "email": p.email}


@app.get("/photographers/{photographer_id}", dependencies=[Depends(require_key)])
def get_photographer(photographer_id: int, db: Session = Depends(get_db)):
    p = db.get(Photographer, photographer_id)
    if not p:
        raise HTTPException(404, "Photographer not found")
    return {
        "id": p.id, "name": p.name, "email": p.email,
        "uploads": [
            {"id": u.id, "filename": u.original_filename, "face_key": u.face_key,
             "thumb_url": f"/uploads/{u.id}/image?size=thumb" if u.thumb_path else None}
            for u in p.uploads
        ],
    }


@app.post("/photographers/{photographer_id}/uploads", status_code=201, dependencies=[Depends(require_key)])
async def upload_photo(
    photographer_id: int,
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL),
    detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True),
    enforce_detection: bool = Form(True),
    database_type: str = Form(DEFAULT_DB),
    db: Session = Depends(get_db),
):
    """Register a photographer's image: store its embeddings in deepface AND an `uploads` row linking to them."""
    if not db.get(Photographer, photographer_id):
        raise HTTPException(404, "Photographer not found")
    img, data = await read_image(file, with_bytes=True)
    original, ext, mime = prepare_original(img, data)

    face_key = f"upl_{uuid.uuid4().hex}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_name, thumb_name = f"{face_key}{ext}", f"{face_key}_thumb.jpg"
    upload = Upload(
        photographer_id=photographer_id, face_key=face_key, original_filename=file.filename,
        image_path=image_name, thumb_path=thumb_name, content_type=mime,
    )
    db.add(upload)
    db.flush()  # row is ready, but nothing is committed until files + embeddings both succeeded
    try:
        (UPLOAD_DIR / image_name).write_bytes(original)
        (UPLOAD_DIR / thumb_name).write_bytes(make_thumb(img))
        result = await run(
            DeepFace.register, img=img, img_name=face_key,
            **common(model_name, detector_backend, align, enforce_detection, database_type),
        )
        if isinstance(result, dict) and result.get("inserted", 1) == 0:
            # deepface de-duplicates identical faces/embeddings, so nothing new was stored
            raise HTTPException(409, "This image was already registered")
        db.commit()
    except Exception:
        db.rollback()
        (UPLOAD_DIR / image_name).unlink(missing_ok=True)   # don't leave files without a database row
        (UPLOAD_DIR / thumb_name).unlink(missing_ok=True)
        raise
    return {
        "upload_id": upload.id, "face_key": face_key, "photographer_id": photographer_id,
        "image_url": f"/uploads/{upload.id}/image", "thumb_url": f"/uploads/{upload.id}/image?size=thumb",
        "deepface": result,
    }


@app.get("/uploads/{upload_id}/image", dependencies=[Depends(require_key)])
def get_upload_image(
    upload_id: int,
    size: str = Query("full", pattern="^(full|thumb)$"),
    db: Session = Depends(get_db),
):
    """Serve the stored original (size=full) or its 320px thumbnail (size=thumb)."""
    u = db.get(Upload, upload_id)
    rel = (u.thumb_path if size == "thumb" else u.image_path) if u else None
    path = (UPLOAD_DIR / rel) if rel else None
    if not path or not path.is_file():
        raise HTTPException(404, "Image not found")
    media = "image/jpeg" if size == "thumb" else (u.content_type or "application/octet-stream")
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


@app.get("/photographers", dependencies=[Depends(require_key)])
def list_photographers(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Photographer, func.count(Upload.id))
        .outerjoin(Upload, Upload.photographer_id == Photographer.id)
        .group_by(Photographer.id)
        .order_by(Photographer.name)
    ).all()
    return [{"id": p.id, "name": p.name, "email": p.email, "upload_count": n} for p, n in rows]


# --------------------------------------------------------------------------- #
# Web UI (same origin as the API, so no CORS setup is needed)
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")