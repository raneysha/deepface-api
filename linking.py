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
from typing import Any, Iterator, Optional

from sqlalchemy import DateTime, ForeignKey, String, create_engine, func, inspect, select, text
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)


def _default_url() -> str:
    uri = os.getenv("DEEPFACE_POSTGRES_URI", "postgresql://deepface:deepface@localhost:5432/deepface")
    return uri.replace("postgresql://", "postgresql+psycopg2://", 1)


DATABASE_URL = os.getenv("APP_DATABASE_URL") or _default_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Photographer(Base):
    __tablename__ = "photographers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[Optional[str]] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
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


def init_db() -> None:
    Base.metadata.create_all(engine)
    # create_all never alters existing tables, so add columns introduced after the first release
    existing = {c["name"] for c in inspect(engine).get_columns("uploads")}
    with engine.begin() as conn:
        for name, ddl in (("image_path", "VARCHAR(500)"), ("thumb_path", "VARCHAR(500)"), ("content_type", "VARCHAR(100)")):
            if name not in existing:
                conn.execute(text(f"ALTER TABLE uploads ADD COLUMN {name} {ddl}"))


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


def attach_owners(matches: Any, db: Session) -> Any:
    """Add `upload` and `photographer` to each search hit (None if there's no linked row)."""
    hits = list(_walk(matches))
    keys = {h["img_name"] for h in hits}
    rows = (
        db.execute(select(Upload, Photographer).join(Photographer).where(Upload.face_key.in_(keys))).all()
        if keys else []
    )
    by_key = {u.face_key: (u, p) for u, p in rows}
    for h in hits:
        pair = by_key.get(h["img_name"])
        if pair:
            u, p = pair
            h["upload"] = {
                "id": u.id, "filename": u.original_filename, "created_at": u.created_at.isoformat(),
                "image_url": f"/uploads/{u.id}/image" if u.image_path else None,
                "thumb_url": f"/uploads/{u.id}/image?size=thumb" if u.thumb_path else None,
            }
            h["photographer"] = {"id": p.id, "name": p.name, "email": p.email}
        else:
            h["upload"] = h["photographer"] = None
    return matches