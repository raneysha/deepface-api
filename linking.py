"""
App-side relational tables (photographers, uploads) + the join back to deepface.

deepface owns its `embeddings` table, so we don't add foreign keys to it.
Instead every upload gets a unique `face_key`. We pass it to DeepFace.register(img_name=face_key),
which makes deepface store it in embeddings.img_name, and every /search hit returns
that img_name. Joining on it gives us the upload and the photographer.

    photographers 1 ──< uploads (face_key)  ···  embeddings.img_name (deepface)
"""
import os
from datetime import datetime
from typing import Any, Callable, Iterator, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, create_engine, func, inspect, select, text
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)


def _default_url() -> str:
    uri = os.getenv("DEEPFACE_POSTGRES_URI", "postgresql://deepface:deepface@localhost:5432/deepface")
    for prefix in ("postgresql://", "postgres://"):   # SQLAlchemy rejects the short "postgres://" form some hosts hand out
        if uri.startswith(prefix):
            return "postgresql+psycopg2://" + uri[len(prefix):]
    return uri


DATABASE_URL = os.getenv("APP_DATABASE_URL") or _default_url()
# Keep the pool small: hosted poolers (Supabase session mode) allow only a handful of connections in total, and DeepFace opens its own.
_POOL = {} if DATABASE_URL.startswith("sqlite") else dict(
    pool_size=int(os.getenv("DB_POOL_SIZE", "3")), max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "2")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "10")), pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "300")))
engine = create_engine(DATABASE_URL, pool_pre_ping=True, **_POOL)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)   # stored lowercase
    full_name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), index=True)                  # admin | photographer | client
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    token_version: Mapped[int] = mapped_column(default=0)                      # bump to invalidate old JWTs
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    photographer: Mapped[Optional["Photographer"]] = relationship(back_populates="user", uselist=False)


class Photographer(Base):
    __tablename__ = "photographers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[Optional[str]] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), unique=True)
    user: Mapped[Optional[User]] = relationship(back_populates="photographer")
    uploads: Mapped[list["Upload"]] = relationship(back_populates="photographer", cascade="all, delete-orphan")


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[int] = mapped_column(primary_key=True)
    photographer_id: Mapped[int] = mapped_column(ForeignKey("photographers.id", ondelete="CASCADE"), index=True)
    face_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # == deepface embeddings.img_name
    original_filename: Mapped[Optional[str]] = mapped_column(String(500))
    image_path: Mapped[Optional[str]] = mapped_column(String(500))   # relative to UPLOAD_DIR
    thumb_path: Mapped[Optional[str]] = mapped_column(String(500))
    content_type: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    photographer: Mapped[Photographer] = relationship(back_populates="uploads")


def get_db() -> Iterator[Session]:
    with SessionLocal() as db:
        yield db


_NEW_COLUMNS = {   # create_all never alters existing tables, so columns added in later versions are added here
    "uploads": [("image_path", "VARCHAR(500)"), ("thumb_path", "VARCHAR(500)"), ("content_type", "VARCHAR(100)")],
    "photographers": [("user_id", "INTEGER REFERENCES users(id) ON DELETE SET NULL")],
}


def init_db() -> None:
    Base.metadata.create_all(engine)
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _NEW_COLUMNS.items():
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    if name == "user_id":
                        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_photographers_user_id ON photographers (user_id)"))


def _walk(o: Any) -> Iterator[dict]:
    """Yield every dict that has an img_name, whatever shape search() returned."""
    if isinstance(o, dict):
        if "img_name" in o:
            yield o
        else:
            for v in o.values():
                yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def best_per_photo(matches: Any) -> Any:
    """
    A group photo is stored as several embeddings (one per face) under the same img_name, so one query face
    can match several of them. Keep only the closest (= highest confidence) match per photo, separately for
    each face found in the query image. Hits without an img_name are left alone.
    """
    def dedupe(hits: list) -> list:
        best: dict = {}
        for i, h in enumerate(hits):
            key = h.get("img_name") if isinstance(h, dict) else None
            key = key if key is not None else ("__unnamed__", i)
            dist = h.get("distance", float("inf")) if isinstance(h, dict) else float("inf")
            if key not in best or dist < best[key].get("distance", float("inf")):
                best[key] = h
        return sorted(best.values(), key=lambda h: h.get("distance", float("inf")))

    if matches and isinstance(matches[0], list):
        return [dedupe(group) for group in matches]
    return dedupe(matches) if isinstance(matches, list) else matches


def _prune(o: Any, keep: set) -> Any:
    if isinstance(o, list):
        return [_prune(v, keep) for v in o if not isinstance(v, dict) or id(v) in keep]
    return o


def attach_owners(matches: Any, db: Session, viewer: "User", sign: Callable[[int, str], str]) -> Any:
    """
    Add upload + photographer info to each search hit and enforce who may see what:
      admin        every hit (full details, including hits with no linked upload)
      photographer only hits from their own uploads (full details)
      client       hits that belong to a photographer; sees photographer name and image links only
    Anything else is removed from the result (fail closed).
    """
    hits = list(_walk(matches))
    keys = {h["img_name"] for h in hits}
    rows = (db.execute(select(Upload, Photographer).join(Photographer).where(Upload.face_key.in_(keys))).all()
            if keys else [])
    by_key = {u.face_key: (u, p) for u, p in rows}
    keep: set = set()
    for h in hits:
        u, p = by_key.get(h["img_name"], (None, None))
        if viewer.role == "admin":
            ok = True
        elif viewer.role == "photographer":
            ok = p is not None and p.user_id == viewer.id
        else:
            ok = p is not None
        if not ok:
            continue
        keep.add(id(h))
        if u is None:   # orphan embedding, admin only
            h["upload"] = h["photographer"] = None
            continue
        urls = {"image_url": sign(u.id, "full") if u.image_path else None,
                "thumb_url": sign(u.id, "thumb") if u.thumb_path else None}
        if viewer.role == "client":
            h["upload"] = {"id": u.id, **urls}
            h["photographer"] = {"name": p.name}
            h.pop("img_name", None)
            h.pop("id", None)
        else:
            h["upload"] = {"id": u.id, "filename": u.original_filename, "created_at": u.created_at.isoformat(), **urls}
            h["photographer"] = {"id": p.id, "name": p.name, "email": p.email}
    return matches if viewer.role == "admin" else _prune(matches, keep)
