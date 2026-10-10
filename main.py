"""
FastAPI wrapper around DeepFace's database-backed API with users, roles and JWT auth.

Roles:  admin        manages users, uploads for anyone, sees every search match, builds the index
        photographer uploads to their own profile; searches only their own photos
        client       searches faces; sees matching photos and the photographer's name
"""
import diagnostics  # noqa: F401  (prints the memory limit before the heavy imports below)

# IMPORTANT: load the Postgres driver BEFORE TensorFlow/OpenCV (imported further down). If it is imported after them, its SSL
# library gets mixed up with theirs and the process crashes ("free(): invalid pointer" / segfault) the first time it
# connects to a database over SSL, which Supabase always requires. Do not move this below the other imports.
try:
    import psycopg2  # noqa: F401
except ImportError:  # not using the postgres backend
    pass

import asyncio
import inspect
import json
import os
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from deepface import DeepFace
import facedata
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from auth_routes import FORBIDDEN, NOT_SIGNED_IN, require_roles, router as auth_router
from linking import Photographer, Upload, User, attach_owners, best_per_photo, engine, get_db, init_db
from security import check_config, decode_token, signed_image_url

DEFAULT_DB = os.getenv("FACE_DB_TYPE", "postgres")        # postgres | mongo | pgvector | weaviate | neo4j ...
DEFAULT_MODEL = os.getenv("FACE_MODEL", "Facenet")
DEFAULT_DETECTOR = os.getenv("FACE_DETECTOR", "opencv")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "storage/uploads"))   # original images + thumbnails
diagnostics.log(f"deepface imported; {diagnostics.memory_summary()}")

D_MODEL = "Face recognition model. Defaults to the FACE_MODEL setting; use the same value for uploads and searches."
D_DETECTOR = "Face detector. Defaults to the FACE_DETECTOR setting; use the same value for uploads and searches."
D_ALIGN = "Straighten faces before comparing (recommended)."
D_ENFORCE = "Reject images where no face is found. Leave on: turning it off stores meaningless embeddings."
D_DB = "Face database backend (defaults to FACE_DB_TYPE). Leave unchanged unless you know you need another."
D_METRIC = "Distance metric: cosine, euclidean or euclidean_l2."
CONNECTION_DETAILS = os.getenv("FACE_DB_CONNECTION")
# Max DeepFace jobs running at once (detection + embedding is heavy). Extra requests wait their turn instead of
# overloading the machine, which matters when a photographer uploads hundreds of photos.
FACE_SLOTS = asyncio.Semaphore(int(os.getenv("FACE_WORKERS", "2")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        check_config()   # refuses to start without a strong JWT_SECRET
        diagnostics.log("JWT_SECRET ok")
        diagnostics.log(f"database: {engine.url.render_as_string(hide_password=True)}")
        init_db()
        diagnostics.log("database tables ready")
    except Exception:
        diagnostics.log("STARTUP FAILED, details below")
        traceback.print_exc()
        sys.stderr.flush()
        raise
    diagnostics.log(f"ready; {diagnostics.memory_summary()}")
    yield


API_DESCRIPTION = """
Face search over photographers' photo libraries, built on [DeepFace](https://github.com/serengil/deepface).

## Getting started
1. Press **Authorize** and sign in with your email (as `username`) and password, or call `POST /auth/login`.
2. Browsers keep the session in a secure cookie. Scripts send `Authorization: Bearer <token>`.
3. Photo links in responses are short-lived and signed, so they can be opened without extra headers.

## Roles
| Role | What they can do |
|---|---|
| **admin** | Manage users, upload for any photographer, see every search match, run the face-data tools |
| **photographer** | Upload to their own profile; search shows only matches from their own photos |
| **client** | Search only; sees matching photos and the photographer's name |
"""
TAGS = [
    {"name": "Authentication", "description": "Sign in, sign up, sign out and change your password."},
    {"name": "Users", "description": "**Admin only.** Create, edit, deactivate and delete accounts."},
    {"name": "Photographers", "description": "Photographer profiles, their photo library, and photo upload. Admins see everyone; photographers only their own."},
    {"name": "Search", "description": "Find a face. What you get back depends on your role."},
    {"name": "Images", "description": "Open stored photos through the short-lived signed links returned by other endpoints."},
    {"name": "Face data (admin)", "description": "**Admin only.** Low-level DeepFace operations."},
    {"name": "System", "description": "Health check."},
]
app = FastAPI(
    title="Face Finder API", version="2.0.0", description=API_DESCRIPTION, openapi_tags=TAGS, lifespan=lifespan,
    swagger_ui_parameters={"docExpansion": "none", "persistAuthorization": True, "displayRequestDuration": True},
)
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
class PhotographerItem(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    user_id: Optional[int] = Field(None, description="Login linked to this profile; null if it has none")
    upload_count: int


class UploadItem(BaseModel):
    id: int
    filename: Optional[str] = None
    thumb_url: Optional[str] = Field(None, description="Signed link, valid for a few minutes")
    image_url: Optional[str] = Field(None, description="Signed link, valid for a few minutes")


class PhotographerDetail(BaseModel):
    id: int
    name: str
    email: Optional[str] = None
    total: int = Field(description="All photos in this profile (the page below holds up to `limit`)")
    uploads: List[UploadItem]


class UploadOut(BaseModel):
    upload_id: int
    face_key: str = Field(description="Internal key linking this photo to its stored face data")
    photographer_id: int
    image_url: str
    thumb_url: str
    deepface: Any = Field(None, description="Raw DeepFace result, e.g. {'inserted': 1}")


class DeleteUserOut(BaseModel):
    deleted_user_id: int
    photographer_profile_deleted: bool
    photos_deleted: int
    photos_kept: int
    face_embeddings_deleted: int


SEARCH_EXAMPLE = {"matches": [[{
    "distance": 0.241, "threshold": 0.4, "confidence": 75.0,
    "photographer": {"id": 1, "name": "Debb", "email": "debb@example.com"},
    "upload": {"id": 12, "filename": "IMG_0412.jpg", "created_at": "2026-10-08T07:30:00",
               "thumb_url": "/uploads/12/image?size=thumb&token=...", "image_url": "/uploads/12/image?size=full&token=..."},
}]]}
ADMIN = Depends(require_roles("admin"))
ANY_ROLE = require_roles("admin", "photographer", "client")
STAFF = require_roles("admin", "photographer")


@app.get("/health", tags=["System"], summary="Health check", description="No sign-in needed.")
def health():
    return {"status": "ok"}


@app.post("/register", tags=["Face data (admin)"], summary="Register faces without a photographer",
          responses={**NOT_SIGNED_IN, **FORBIDDEN, 422: {"description": "No face found in an image"}},
          description="**Admin only.** Stores faces in DeepFace's database with no photographer or stored photo attached. "
                      "For normal use upload through a photographer profile instead.")
async def register(
    files: List[UploadFile] = File(..., description="One or more face images (not linked to a photographer)"),
    model_name: str = Form(DEFAULT_MODEL, description=D_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR, description=D_DETECTOR),
    align: bool = Form(True, description=D_ALIGN), enforce_detection: bool = Form(True, description=D_ENFORCE), database_type: str = Form(DEFAULT_DB, description=D_DB),
    _: User = ADMIN,
):
    imgs = [await read_image(f) for f in files]
    result = await run(DeepFace.register, img=imgs if len(imgs) > 1 else imgs[0],
                       **common(model_name, detector_backend, align, enforce_detection, database_type))
    return {"registered_files": [f.filename for f in files], "result": result}


@app.post("/index", tags=["Face data (admin)"], summary="Rebuild the fast-search index",
          responses={**NOT_SIGNED_IN, **FORBIDDEN},
          description="**Admin only.** Builds the lookup structure used by approximate search (`ann=true` on `/search`). "
                      "Not needed for normal search. Requires the optional `faiss-cpu` package. Re-run after adding many photos.")
async def build_index(
    model_name: str = Form(DEFAULT_MODEL, description=D_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR, description=D_DETECTOR),
    align: bool = Form(True, description=D_ALIGN), database_type: str = Form(DEFAULT_DB, description=D_DB), _: User = ADMIN,
):
    """Build the ANN index (FAISS); re-run after adding identities."""
    await run(DeepFace.build_index, model_name=model_name, detector_backend=detector_backend, align=align,
              database_type=database_type, connection_details=CONNECTION_DETAILS)
    return {"status": "index built"}


@app.post("/search", tags=["Search"], summary="Search for a face",
          responses={200: {"content": {"application/json": {"example": SEARCH_EXAMPLE}}}, **NOT_SIGNED_IN,
                     422: {"description": "No face found in the image"}},
          description="Upload a photo; get the closest stored faces, **one best match per photo**. `matches` holds one list per face found in "
                      "your photo. What you see depends on your role: admins see everything, photographers only their own photos, "
                      "clients the matching photos with the photographer's name.")
async def search(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL, description=D_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR, description=D_DETECTOR),
    distance_metric: str = Form("cosine", description=D_METRIC), align: bool = Form(True, description=D_ALIGN), enforce_detection: bool = Form(True, description=D_ENFORCE),
    ann: bool = Form(False, description="Approximate search (admin builds the index first)"),
    database_type: str = Form(DEFAULT_DB, description=D_DB),
    user: User = Depends(ANY_ROLE), db: Session = Depends(get_db),
):
    """1:N face search. Results are filtered by the caller's role (see attach_owners)."""
    img = await read_image(file)
    # ANN returns only the top k (default 20); fetch extra candidates since duplicates per photo are merged below
    extra = {"search_method": "ann", "search_type": "ann", "k": 50} if ann else {}   # unsupported names are dropped
    matches = await run(DeepFace.search, img=img, distance_metric=distance_metric, **extra,
                        **common(model_name, detector_backend, align, enforce_detection, database_type))
    matches = attach_owners(best_per_photo(matches), db, viewer=user, sign=lambda upload_id, size: signed_image_url(upload_id, user.id, size))
    return {"matches": matches}


# ----- photographers & uploads ---------------------------------------------------------------------------------
def photographer_for(db: Session, user: User, pid: int) -> Photographer:
    p = db.get(Photographer, pid)
    if not p or (user.role != "admin" and p.user_id != user.id):
        raise HTTPException(404, "Photographer not found")   # 404 rather than 403: don't reveal other profiles exist
    return p


@app.get("/photographers", tags=["Photographers"], summary="List photographer profiles", response_model=List[PhotographerItem],
         responses={**NOT_SIGNED_IN, **FORBIDDEN}, description="Admins get every profile; photographers only their own.")
def list_photographers(user: User = Depends(STAFF), db: Session = Depends(get_db)):
    q = (select(Photographer, func.count(Upload.id))
         .outerjoin(Upload, Upload.photographer_id == Photographer.id)
         .group_by(Photographer.id).order_by(Photographer.name))
    if user.role != "admin":
        q = q.where(Photographer.user_id == user.id)
    return [{"id": p.id, "name": p.name, "email": p.email, "user_id": p.user_id, "upload_count": n}
            for p, n in db.execute(q).all()]


@app.get("/photographers/{photographer_id}", tags=["Photographers"], summary="Get a profile and its photos",
         response_model=PhotographerDetail, responses={**NOT_SIGNED_IN, **FORBIDDEN, 404: {"description": "Not found, or not yours"}},
         description="Newest photos first, in pages (`limit`/`offset`). Photo links are signed and short-lived.")
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


@app.post("/photographers/{photographer_id}/uploads", status_code=201, tags=["Photographers"], summary="Upload a photo",
          response_model=UploadOut,
          responses={**NOT_SIGNED_IN, **FORBIDDEN, 400: {"description": "Not a readable image"}, 404: {"description": "Profile not found, or not yours"},
                     409: {"description": "This image was already registered"}, 413: {"description": "Image larger than MAX_UPLOAD_MB"},
                     422: {"description": "No face found in the image"}},
          description="Saves the original and a thumbnail, and stores the face data. Upload one image per request (the web page sends "
                      "big batches this way). Admins can upload for any profile, photographers only their own.")
async def upload_photo(
    photographer_id: int, file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL, description=D_MODEL), detector_backend: str = Form(DEFAULT_DETECTOR, description=D_DETECTOR),
    align: bool = Form(True, description=D_ALIGN), enforce_detection: bool = Form(True, description=D_ENFORCE), database_type: str = Form(DEFAULT_DB, description=D_DB),
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


@app.get("/uploads/{upload_id}/image", tags=["Images"], summary="Download a photo", response_class=FileResponse,
         responses={200: {"content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}}, "description": "The image file"},
                    403: {"description": "Invalid or expired link"}, 404: {"description": "Image not found"}},
         description="Needs the signed `token` from a link returned by another endpoint (valid about 15 minutes). No sign-in header required.")
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


@app.delete(
    "/users/{user_id}", tags=["Users"], summary="Delete a user", response_model=DeleteUserOut,
    responses={**NOT_SIGNED_IN, **FORBIDDEN, 400: {"description": "You can't delete your own account"}, 404: {"description": "User not found"},
               501: {"description": "`delete_photos` isn't supported for this face database backend; nothing was deleted"}},
    description="**Admin only.** Permanently removes the account. The user is signed out immediately.\n\n"
                "For a photographer, the default keeps their profile and photos in the library (the profile just has no login, and can be "
                "linked to a new one). Set `delete_photos=true` to also erase the profile, every photo and thumbnail on disk, and the face "
                "data stored for them. That cannot be undone. If you use approximate search, run `POST /index` afterwards.",
)
def delete_user(
    user_id: int,
    delete_photos: bool = Query(False, description="Also delete the photographer profile, all its photos, and their stored face data"),
    admin: User = ADMIN, db: Session = Depends(get_db),
):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id:
        raise HTTPException(400, "You can't delete your own account")
    profile = user.photographer
    uploads = list(profile.uploads) if (profile and delete_photos) else []
    if uploads and not facedata.supported(DEFAULT_DB):
        raise HTTPException(501, f"Deleting face data is only implemented for the postgres backend, not '{DEFAULT_DB}'. Nothing was deleted.")
    files = [UPLOAD_DIR / p for u in uploads for p in (u.image_path, u.thumb_path) if p]
    # Biometric data first: if the account rows fail afterwards, nothing searchable is left behind and the call can simply be retried.
    removed = facedata.delete_face_data([u.face_key for u in uploads], DEFAULT_DB) if uploads else 0
    kept = 0
    if profile and delete_photos:
        db.delete(profile)                    # cascades to its upload rows
    elif profile:
        profile.user_id = None                # keep the photos, drop only the login
        kept = db.scalar(select(func.count(Upload.id)).where(Upload.photographer_id == profile.id))
    db.delete(user)
    db.commit()
    for f in files:                           # after the commit: a leftover file is harmless, a missing one would not be
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    return {"deleted_user_id": user_id, "photographer_profile_deleted": bool(profile and delete_photos),
            "photos_deleted": len(uploads), "photos_kept": kept, "face_embeddings_deleted": removed}


@app.get("/delete-account", include_in_schema=False)
def delete_account_page():
    """Public page where a client can delete their account without the app (required by Google Play)."""
    return FileResponse(Path(__file__).parent / "static" / "delete-account.html")


@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(Path(__file__).parent / "static" / "index.html")