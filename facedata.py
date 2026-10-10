"""Erasing face data that deepface stores (it has no delete API of its own). Postgres backend only."""
from typing import Iterable, Optional

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from linking import _default_url

SUPPORTED_BACKENDS = ("postgres",)
_engine: Optional[Engine] = None


def supported(database_type: str) -> bool:
    return database_type in SUPPORTED_BACKENDS


def face_engine() -> Engine:
    """Connection to the database deepface writes to (same URI deepface reads: DEEPFACE_POSTGRES_URI)."""
    global _engine
    if _engine is None:
        _engine = create_engine(_default_url(), pool_pre_ping=True)
    return _engine


def delete_face_data(face_keys: Iterable[str], database_type: str) -> int:
    """Delete the embeddings stored under these img_names. Returns how many embedding rows were removed."""
    keys = list(face_keys)
    if not supported(database_type):
        raise NotImplementedError(f"Deleting face data is not implemented for the '{database_type}' backend")
    if not keys:
        return 0
    stmt = text("DELETE FROM embeddings WHERE img_name IN :keys").bindparams(bindparam("keys", expanding=True))
    removed = 0
    with face_engine().begin() as conn:
        for i in range(0, len(keys), 500):
            removed += conn.execute(stmt, {"keys": keys[i:i + 500]}).rowcount
    # The approximate-search index (a FAISS blob) still contains the deleted vectors, so drop it too.
    # Approximate search needs POST /index again afterwards; exact search is unaffected.
    try:
        with face_engine().begin() as conn:
            conn.execute(text("DELETE FROM embeddings_index"))
    except SQLAlchemyError:
        pass   # table missing: the index was never built
    return removed
