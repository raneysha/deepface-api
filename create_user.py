"""Create the first admin (or any user) from the command line:  python create_user.py --email you@x.com --name "You" --role admin"""
import argparse
import getpass
import sys

from dotenv import load_dotenv

load_dotenv()  # read .env before linking.py builds the DB engine

from sqlalchemy import select  # noqa: E402

from linking import Photographer, SessionLocal, User, init_db  # noqa: E402
from security import ROLES, hash_password, validate_password  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--email", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--role", choices=ROLES, default="admin")
args = ap.parse_args()

pw = getpass.getpass("Password (10+ chars): ")
if pw != getpass.getpass("Repeat password: "):
    sys.exit("Passwords do not match")
try:
    validate_password(pw)
except ValueError as e:
    sys.exit(str(e))

init_db()
email = args.email.strip().lower()
with SessionLocal() as db:
    if db.scalar(select(User).where(User.email == email)):
        sys.exit("A user with that email already exists")
    user = User(email=email, full_name=args.name, role=args.role, password_hash=hash_password(pw))
    db.add(user)
    db.flush()
    if args.role == "photographer":
        db.add(Photographer(name=args.name, email=email, user_id=user.id))
    db.commit()
    print(f"Created {args.role} {email} (id {user.id})")
