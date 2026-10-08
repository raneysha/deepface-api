# DeepFace DB API

FastAPI service + web UI on top of DeepFace's database-backed face recognition.
Photographers and their uploads live in relational tables; face embeddings live in
DeepFace's own `embeddings` table; they are joined via `uploads.face_key == embeddings.img_name`.

## Setup

Requirements: Python 3.10 or 3.11, Docker (for Postgres).

```bash
# 1. Postgres
docker compose up -d

# 2. Python env
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env                 # edit API_KEY at least

# 4. Run
uvicorn main:app --env-file .env --port 8000
```

- Web UI:  http://localhost:8000
- API docs: http://localhost:8000/docs

First upload/search is slow: DeepFace downloads model weights (~100+ MB) on first use.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Web UI |
| `GET/POST /photographers`, `GET /photographers/{id}` | Photographers |
| `POST /photographers/{id}/uploads` | Register an image and link it to the photographer |
| `POST /search` | 1:N face search; adds `upload` + `photographer` to every match |
| `POST /register` | Register an image with no photographer link |
| `POST /identify` | 1:1 check (needs a deepface version that has `identify`) |
| `POST /index` | Build the ANN index (`ann=true` in `/search`) |

All routes except `/`, `/health`, `/docs` need `X-API-Key` when `API_KEY` is set.

## Where images are stored

Each upload's original image and a 320px thumbnail are saved to `UPLOAD_DIR` (default `storage/uploads/`),
named after the upload's `face_key` (`upl_<id>.jpg|png|...` and `upl_<id>_thumb.jpg`). The `uploads` table
records the paths. Images are served by `GET /uploads/{id}/image` (`?size=thumb` for the thumbnail) and
require the `X-API-Key` header, so the web UI fetches them with `fetch()` rather than plain `<img src>`.

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
# deepface-api
