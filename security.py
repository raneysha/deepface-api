"""Password hashing (Argon2) and JWTs (access tokens + short-lived signed image links)."""
import os
import time

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

ROLES = ("admin", "photographer", "client")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALG = "HS256"
ACCESS_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "480"))
IMAGE_URL_MINUTES = int(os.getenv("IMAGE_URL_MINUTES", "15"))
COOKIE_NAME = "session"
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"   # set true behind HTTPS
ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "true").lower() == "true"       # public sign-up creates clients only


def check_config() -> None:
    if len(JWT_SECRET) < 32:
        raise RuntimeError('Set JWT_SECRET to 32+ random characters, e.g. '
                           'python -c "import secrets; print(secrets.token_urlsafe(48))"')


_ph = PasswordHasher()
_DUMMY_HASH = _ph.hash("dummy-password-used-for-timing")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, password)
    except (VerificationError, InvalidHashError):
        return False


def burn_time(password: str) -> None:
    """Hash-check against a dummy so unknown emails cost the same time as wrong passwords."""
    verify_password(password, _DUMMY_HASH)


def validate_password(password: str) -> None:
    if not 10 <= len(password) <= 128:
        raise ValueError("Password must be 10 to 128 characters")


def _encode(claims: dict, minutes: int) -> str:
    now = int(time.time())
    return jwt.encode({**claims, "iat": now, "exp": now + minutes * 60}, JWT_SECRET, algorithm=JWT_ALG)


def create_access_token(user) -> str:
    # `tv` lets us invalidate every old token by bumping user.token_version (password change, role change, ...)
    return _encode({"sub": str(user.id), "tv": user.token_version, "typ": "access"}, ACCESS_MINUTES)


def decode_token(token: str, typ: str):
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG], options={"require": ["exp", "sub", "typ"]})
    except jwt.PyJWTError:
        return None
    return data if data.get("typ") == typ else None   # an access token can't be used as an image link, or vice versa


def signed_image_url(upload_id: int, user_id: int, size: str = "full") -> str:
    token = _encode({"sub": str(user_id), "upl": upload_id, "typ": "img"}, IMAGE_URL_MINUTES)
    return f"/uploads/{upload_id}/image?size={size}&token={token}"
