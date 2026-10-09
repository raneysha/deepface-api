"""Login, sign-up, current-user dependencies and admin user management."""
import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from linking import Photographer, User, get_db
from security import (ACCESS_MINUTES, ALLOW_SIGNUP, COOKIE_NAME, COOKIE_SECURE, burn_time, create_access_token,
                      decode_token, hash_password, validate_password, verify_password)

router = APIRouter()
oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)   # also powers the Authorize button in /docs
Role = Literal["admin", "photographer", "client"]
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


def current_user(request: Request, bearer: Optional[str] = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    """Accepts `Authorization: Bearer <jwt>` (API clients) or the httpOnly session cookie (web UI)."""
    token, via_cookie = (bearer, False) if bearer else (request.cookies.get(COOKIE_NAME), True)
    claims = decode_token(token, "access") if token else None
    if not claims:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    # CSRF defence for cookie sessions: browsers can't add this header cross-site without a CORS preflight (we allow none)
    if via_cookie and request.method not in SAFE_METHODS and request.headers.get("x-requested-with") != "fetch":
        raise HTTPException(403, "Missing X-Requested-With: fetch header")
    user = db.get(User, int(claims["sub"]))
    # role, active flag and token version are re-read from the DB on every request, so changes apply immediately
    if not user or not user.is_active or user.token_version != claims.get("tv"):
        raise HTTPException(401, "Session expired, please sign in again", headers={"WWW-Authenticate": "Bearer"})
    return user


def require_roles(*roles: str):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "Your role is not allowed to do this")
        return user
    return dep


def user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role, "is_active": u.is_active,
            "photographer_id": u.photographer.id if u.photographer else None}


def start_session(response: Response, user: User) -> dict:
    token = create_access_token(user)
    response.set_cookie(COOKIE_NAME, token, max_age=ACCESS_MINUTES * 60, httponly=True,
                        samesite="strict", secure=COOKIE_SECURE, path="/")
    return {"access_token": token, "token_type": "bearer", "user": user_out(user)}


def check_password(pw: str) -> None:
    try:
        validate_password(pw)
    except ValueError as e:
        raise HTTPException(422, str(e))


def ensure_profile(db: Session, user: User, link_id: Optional[int] = None) -> None:
    """A photographer user needs a photographer profile: link an unlinked existing one, or create one."""
    if user.photographer:
        return
    if link_id is not None:
        p = db.get(Photographer, link_id)
        if not p or p.user_id is not None:
            raise HTTPException(422, "That photographer profile does not exist or already has a login")
        p.user_id = user.id
    else:
        db.add(Photographer(name=user.full_name, email=user.email, user_id=user.id))


# ----- login / sign-up ------------------------------------------------------------------------------------------
_fails: dict = {}   # in-memory login throttle: fine for one process, use Redis or a proxy limit if you scale out


def _throttled(key: str) -> bool:
    now = time.time()
    _fails[key] = [t for t in _fails.get(key, []) if now - t < 900]
    return len(_fails[key]) >= 5


@router.post("/auth/login")
def login(request: Request, response: Response, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    email = form.username.strip().lower()
    key = f"{email}|{request.client.host if request.client else ''}"
    if _throttled(key):
        raise HTTPException(429, "Too many failed attempts. Try again in 15 minutes.")
    user = db.scalar(select(User).where(User.email == email))
    ok = verify_password(form.password, user.password_hash) if user else (burn_time(form.password) or False)
    if not (ok and user.is_active):
        _fails.setdefault(key, []).append(time.time())
        raise HTTPException(401, "Incorrect email or password")
    _fails.pop(key, None)
    return start_session(response, user)


class SignupIn(BaseModel):
    email: EmailStr
    password: str
    full_name: str = Field(min_length=1, max_length=200)


@router.post("/auth/signup", status_code=201)
def signup(body: SignupIn, response: Response, db: Session = Depends(get_db)):
    """Public sign-up always creates a *client*. Photographers and admins are created by an admin."""
    if not ALLOW_SIGNUP:
        raise HTTPException(403, "Sign-up is disabled. Ask an administrator for an account.")
    check_password(body.password)
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "An account with this email already exists")
    user = User(email=email, full_name=body.full_name.strip(), role="client", password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return start_session(response, user)


@router.post("/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"status": "signed out"}


@router.get("/auth/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@router.post("/auth/change-password")
def change_password(body: PasswordChange, response: Response, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    check_password(body.new_password)
    user.password_hash = hash_password(body.new_password)
    user.token_version += 1   # signs out every other session
    db.commit()
    return start_session(response, user)


# ----- admin: user management -----------------------------------------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str = Field(min_length=1, max_length=200)
    role: Role
    photographer_id: Optional[int] = None   # link an existing photographer profile that has no login yet


class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None          # admin password reset


@router.get("/users")
def list_users(_: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    return [user_out(u) for u in db.scalars(select(User).order_by(User.full_name))]


@router.post("/users", status_code=201)
def create_user(body: UserCreate, _: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    check_password(body.password)
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "An account with this email already exists")
    user = User(email=email, full_name=body.full_name.strip(), role=body.role, password_hash=hash_password(body.password))
    db.add(user)
    db.flush()
    if body.role == "photographer":
        ensure_profile(db, user, body.photographer_id)
    db.commit()
    db.refresh(user)   # reload relationships (the photographer profile may have just been created/linked)
    return user_out(user)


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserUpdate, admin: User = Depends(require_roles("admin")), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    if user.id == admin.id and ((body.role and body.role != "admin") or body.is_active is False):
        raise HTTPException(400, "You can't demote or deactivate your own account")
    if body.full_name:
        user.full_name = body.full_name.strip()
    if body.role and body.role != user.role:
        user.role = body.role
        user.token_version += 1
        if body.role == "photographer":
            ensure_profile(db, user)
    if body.is_active is not None and body.is_active != user.is_active:
        user.is_active = body.is_active
        user.token_version += 1
    if body.password:
        check_password(body.password)
        user.password_hash = hash_password(body.password)
        user.token_version += 1
    db.commit()
    db.refresh(user)   # reload relationships (the photographer profile may have just been created/linked)
    return user_out(user)
