# DeepFace DB API

FastAPI service + web UI on top of DeepFace's database-backed face recognition.
Photographers and their uploads live in relational tables; face embeddings live in
DeepFace's own `embeddings` table; they are joined via `uploads.face_key == embeddings.img_name`.

## Setup

Requirements: Python 3.10+ and Docker (for Postgres; a native or hosted Postgres works too).

```bash
docker compose up -d                       # Postgres
python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                       # Windows: copy .env.example .env
```

Edit `.env` and set `JWT_SECRET` (the app refuses to start without it):

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Create the first admin (prompts for a password, never stored in `.env`):

```bash
python create_user.py --email you@example.com --name "Your Name" --role admin
```

Run:

```bash
uvicorn main:app --env-file .env --port 8000
```

Open http://localhost:8000 and sign in. API docs: http://localhost:8000/docs (use the Authorize button).
First upload/search is slow: DeepFace downloads model weights on first use.

## Roles

| | admin | photographer | client |
|---|---|---|---|
| Create/manage users, change roles, deactivate | yes | | |
| Upload photos | for any photographer | own profile only | |
| Search | all matches, full details | only matches from own photos | matches linked to a photographer; sees photographer name + photo only |
| Build index, `/register` | yes | | |

Public sign-up (`POST /auth/signup`) always creates a **client**. Photographers and admins are created by an
admin (`POST /users`, or the Users panel). Setting `ALLOW_SIGNUP=false` turns public sign-up off.
Upgrading from the previous version: existing photographer profiles keep working. In the Users panel, add a
photographer user and pick "Link existing: <name>" to attach a login to an old profile.

## Authentication

- Passwords are hashed with Argon2 (10+ characters required).
- Login (`POST /auth/login`, form fields `username` = email, `password`) returns a JWT and also sets it as an
  `httpOnly`, `SameSite=Strict` cookie, which the web UI uses. API clients send `Authorization: Bearer <token>`.
- Role, active flag and a token version are re-checked in the database on every request, so deactivating a
  user, changing a role or changing a password takes effect immediately and signs out old sessions.
- Cookie sessions must send `X-Requested-With: fetch` on POST/PATCH (CSRF guard). The web UI does this.
- Photo URLs are short-lived signed links (`IMAGE_URL_MINUTES`) issued only for photos the user may see, so
  upload ids can't be enumerated. Anyone holding a link can open that photo until it expires.
- Login is throttled (5 failures / 15 min per email+IP, in memory: use a proxy rate limit if you run several processes).
- Set `COOKIE_SECURE=true` and serve over HTTPS in production.

## Mobile app (clients)

`../facefinder-android` is an Android app for **client** accounts (sign up, sign in, search). It uses the same API:
`/auth/signup`, `/auth/login`, `/search`, `/auth/delete-account`, and the signed photo links. It needs `ALLOW_SIGNUP=true`,
and a deployed server must be reachable over HTTPS (Android release builds block plain http).

Clients can delete their own account (`POST /auth/delete-account`, password required), and there is a public web page for it at
`/delete-account`. Google Play requires both for apps that create accounts. Photographer and admin accounts can't be
self-deleted, because they own photos and face data; an admin removes them (see below).

## Deleting users

`DELETE /users/{id}` (admin only, not your own account) removes the account and signs the user out at once.

- **Default:** a photographer's profile and photos stay in the library, only the login is removed. The profile can be
  given a new login later (create a user with `photographer_id`).
- **`?delete_photos=true`:** also erases the profile, every photo and thumbnail on disk, and the face data stored for
  them. Cannot be undone. In the web page this is the "Delete account and N photos" button.
- Face data erasure is implemented for the **postgres** backend only. For other backends the request is refused
  with 501 and nothing is deleted. Erasing also drops the approximate-search index (`embeddings_index`), because it
  still holds the deleted vectors; run `POST /index` again if you use approximate search.

## API docs

Interactive docs at `/docs`, grouped by function: Authentication, Users, Photographers, Search, Images,
Face data (admin), System. Each endpoint says who can call it and lists its error responses.

## Endpoints

| Route | Who | Purpose |
|---|---|---|
| `POST /auth/login`, `/auth/logout`, `/auth/signup`, `/auth/change-password`, `/auth/delete-account` (clients), `GET /auth/me` | see above | Session handling |
| `GET/POST /users`, `PATCH/DELETE /users/{id}` | admin | Manage users (see Deleting users) |
| `GET /photographers`, `GET /photographers/{id}` | admin, photographer (own) | Profiles and gallery |
| `POST /photographers/{id}/uploads` | admin, photographer (own) | Register an image and link it |
| `POST /search` | all roles | 1:N face search, filtered by role |
| `POST /index`, `/register` | admin | Index and low-level deepface calls |
| `GET /uploads/{id}/image?token=...` | signed link | Original or `?size=thumb` |

## Web interface

The page is organised in tabs: **Library** (photographers, upload, photo gallery), **Search** and, for admins,
**Users**. Photographers get Library and Search; clients only see Search. The selected tab is kept in the URL
(`#library`, `#search`, `#users`) so a refresh stays on it, and the tabs work with the arrow keys. An upload keeps
running if you switch tabs (the Library tab shows "uploading…"), but closing the page stops it.

## Mass upload

Photographers (and admins) can drop photos or whole folders onto the upload area, or use "Choose photos" /
"Choose a folder". Each photo is sent as its own request, a few at a time, with a progress bar and a summary:
added, already registered (duplicates are skipped), no face found, and failed (with the reason). "Retry failed"
re-sends only the failures. If the session expires mid-batch, the remaining photos stay queued and nothing
already uploaded is lost. Only JPG, PNG, WebP and BMP are supported (HEIC from iPhones must be exported as JPG first).

Server-side limits: `FACE_WORKERS` (default 2) caps concurrent face-detection jobs so a big batch can't
overload the machine; extra requests wait their turn. `MAX_UPLOAD_MB` (default 10) is the per-image size limit.
The photo library is paged (60 per page, "Load more").

## Where images are stored

Each upload's original image and a 320px thumbnail are saved to `UPLOAD_DIR` (default `storage/uploads/`),
named after the upload's `face_key` (`upl_<id>.jpg|png|...` and `upl_<id>_thumb.jpg`). The `uploads` table
records the paths. Images are served by `GET /uploads/{id}/image` (`?size=thumb` for the thumbnail) and
need a signed `token` link (see Authentication).

- Back up `storage/uploads/` together with the database; they must stay in sync.
- Local disk only works for a single server. For several instances, mount shared storage or move to S3.
- Upgrading from an earlier version: new columns on `uploads` are added automatically at startup.
  Uploads made before this version have no stored image (the UI simply shows no thumbnail for them).

## Notes

- Deleting rows from `uploads`/`photographers` does not delete embeddings or image files. To clean up:
  `DELETE FROM embeddings WHERE img_name = '<face_key>';`, remove the files for that `face_key` from
  `UPLOAD_DIR`, then `POST /index` if you use ANN.
- Identical images are de-duplicated by DeepFace; re-uploading returns 409.
- Face embeddings are biometric data: use HTTPS, real auth, and check consent/retention rules.
- Using another DB (mongo, pgvector, ...): set `FACE_DB_TYPE`, install its client library, and set
  the connection variable DeepFace expects for it.

## Deploying

See `DEPLOY_RAILWAY.md` (Dockerfile, `railway.json` and a step-by-step guide for Railway).

## Hosted Postgres (Supabase etc.)

Supabase's direct host `db.<ref>.supabase.co` is IPv6-only and fails with "could not translate host name" on
networks without IPv6. Use the **Session pooler** string from Connect in the dashboard (port 5432, user
`postgres.<project-ref>`, host `...pooler.supabase.com`) as `DEEPFACE_POSTGRES_URI`. URL-encode special
characters in the password. You don't need the Docker Postgres in that case.
