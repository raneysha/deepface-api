"""
FastAPI wrapper around DeepFace's database-backed API with users, roles and JWT auth.

Roles:  admin        manages users, uploads for anyone, sees every search match, builds the index
        photographer uploads to their own profile; searches only their own photos
        client       searches faces; sees matching photos and the photographer's name
"""
import asyncio
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
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from auth_routes import require_roles, router as auth_router
from linking import Photographer, Upload, User, attach_owners, get_db, init_db
from security import check_config, decode_token, signed_image_url

DEFAULT_DB = os.getenv("FACE_DB_TYPE", "postgres")        # postgres | mongo | pgvector | weaviate | neo4j ...
DEFAULT_MODEL = os.getenv("FACE_MODEL", "Facenet")
DEFAULT_DETECTOR = os.getenv("FACE_DETECTOR", "opencv")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "storage/uploads"))   # original images + thumbnails
CONNECTION_DETAILS = os.getenv("FACE_DB_CONNECTION")
# Max DeepFace jobs running at once (detection + embedding is heavy). Extra requests wait their turn instead of
# overloading the machine, which matters when a photographer uploads hundreds of photos.
FACE_SLOTS = asyncio.Semaphore(int(os.getenv("FACE_WORKERS", "2")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_config()   # refuses to start without a strong JWT_SECRET
    init_db()
    yield


app = FastAPI(title="DeepFace DB API", version="2.0.0", lifespan=lifespan)
app.include_router(auth_router)


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
        async with FACE_SLOTS:
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
ADMIN = Depends(require_roles("admin"))
ANY_ROLE = require_roles("admin", "photographer", "client")
STAFF = require_roles("admin", "photographer")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/register")
async def register(
    files: List[UploadFile] = File(..., description="One or more face images (not linked to a photographer)"),
    model_name: str = Form(DEFAULT_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True), enforce_detection: bool = Form(True), database_type: str = Form(DEFAULT_DB),
    _: User = ADMIN,
):
    imgs = [await read_image(f) for f in files]
    result = await run(DeepFace.register, img=imgs if len(imgs) > 1 else imgs[0],
                       **common(model_name, detector_backend, align, enforce_detection, database_type))
    return {"registered_files": [f.filename for f in files], "result": result}


@app.post("/identify")
async def identify(
    file: UploadFile = File(...), identity_id: str = Form(...),
    model_name: str = Form(DEFAULT_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True), enforce_detection: bool = Form(True), database_type: str = Form(DEFAULT_DB),
    _: User = ADMIN,
):
    if not hasattr(DeepFace, "identify"):
        raise HTTPException(501, "Installed deepface has no identify(); install from the GitHub master branch.")
    img = await read_image(file)
    return await run(DeepFace.identify, img=img, identity_id=identity_id,
                     **common(model_name, detector_backend, align, enforce_detection, database_type))


@app.post("/index")
async def build_index(
    model_name: str = Form(DEFAULT_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True), database_type: str = Form(DEFAULT_DB), _: User = ADMIN,
):
    """Build the ANN index (FAISS); re-run after adding identities."""
    await run(DeepFace.build_index, model_name=model_name, detector_backend=detector_backend, align=align,
              database_type=database_type, connection_details=CONNECTION_DETAILS)
    return {"status": "index built"}


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR),
    distance_metric: str = Form("cosine"), align: bool = Form(True), enforce_detection: bool = Form(True),
    ann: bool = Form(False, description="Approximate search (admin builds the index first)"),
    database_type: str = Form(DEFAULT_DB),
    user: User = Depends(ANY_ROLE), db: Session = Depends(get_db),
):
    """1:N face search. Results are filtered by the caller's role (see attach_owners)."""
    img = await read_image(file)
    extra = {"search_method": "ann", "search_type": "ann"} if ann else {}   # unsupported name is dropped
    matches = await run(DeepFace.search, img=img, distance_metric=distance_metric, **extra,
                        **common(model_name, detector_backend, align, enforce_detection, database_type))
    matches = attach_owners(matches, db, viewer=user, sign=lambda upload_id, size: signed_image_url(upload_id, user.id, size))
    return {"matches": matches}


# ----- photographers & uploads ---------------------------------------------------------------------------------
def photographer_for(db: Session, user: User, pid: int) -> Photographer:
    p = db.get(Photographer, pid)
    if not p or (user.role != "admin" and p.user_id != user.id):
        raise HTTPException(404, "Photographer not found")   # 404 rather than 403: don't reveal other profiles exist
    return p


@app.get("/photographers")
def list_photographers(user: User = Depends(STAFF), db: Session = Depends(get_db)):
    q = (select(Photographer, func.count(Upload.id))
         .outerjoin(Upload, Upload.photographer_id == Photographer.id)
         .group_by(Photographer.id).order_by(Photographer.name))
    if user.role != "admin":
        q = q.where(Photographer.user_id == user.id)
    return [{"id": p.id, "name": p.name, "email": p.email, "user_id": p.user_id, "upload_count": n}
            for p, n in db.execute(q).all()]


@app.get("/photographers/{photographer_id}")
def get_photographer(
    photographer_id: int, limit: int = Query(60, ge=1, le=200), offset: int = Query(0, ge=0),
    user: User = Depends(STAFF), db: Session = Depends(get_db),
):
    """Profile plus one page of uploads (newest first); `total` is the full count."""
    p = photographer_for(db, user, photographer_id)
    total = db.scalar(select(func.count(Upload.id)).where(Upload.photographer_id == p.id))
    rows = db.scalars(select(Upload).where(Upload.photographer_id == p.id)
                      .order_by(Upload.id.desc()).limit(limit).offset(offset)).all()
    return {"id": p.id, "name": p.name, "email": p.email, "total": total, "uploads": [
        {"id": u.id, "filename": u.original_filename,
         "thumb_url": signed_image_url(u.id, user.id, "thumb") if u.thumb_path else None,
         "image_url": signed_image_url(u.id, user.id, "full") if u.image_path else None}
        for u in rows]}


@app.post("/photographers/{photographer_id}/uploads", status_code=201)
async def upload_photo(
    photographer_id: int, file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR),
    align: bool = Form(True), enforce_detection: bool = Form(True), database_type: str = Form(DEFAULT_DB),
    user: User = Depends(STAFF), db: Session = Depends(get_db),
):
    """Register an image: embeddings go to deepface, the original + thumbnail to disk, and a linked `uploads` row."""
    photographer_for(db, user, photographer_id)
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
        "image_url": signed_image_url(upload.id, user.id, "full"),
        "thumb_url": signed_image_url(upload.id, user.id, "thumb"),
        "deepface": result,
    }


@app.get("/uploads/{upload_id}/image")
def get_upload_image(
    upload_id: int, token: str = Query(..., description="Short-lived signed link from an API response"),
    size: str = Query("full", pattern="^(full|thumb)$"), db: Session = Depends(get_db),
):
    """Images are only reachable through signed, expiring links, so upload ids can't be enumerated."""
    claims = decode_token(token, "img")
    if not claims or claims.get("upl") != upload_id:
        raise HTTPException(403, "Invalid or expired image link")
    u = db.get(Upload, upload_id)
    rel = (u.thumb_path if size == "thumb" else u.image_path) if u else None
    path = (UPLOAD_DIR / rel) if rel else None
    if not path or not path.is_file():
        raise HTTPException(404, "Image not found")
    media = "image/jpeg" if size == "thumb" else (u.content_type or "application/octet-stream")
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=300"})


@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")