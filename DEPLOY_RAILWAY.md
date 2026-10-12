# Deploying to Railway

What you get: the API and web page at `https://<your-app>.up.railway.app` (HTTPS included), photos and downloaded
models on a persistent volume, and your database on Supabase.

Files used: `Dockerfile`, `railway.json` (health check, restart policy), `constraints.txt` (exact tested package versions).

## Before you start

- A Railway account on the **Hobby plan or above**. The free plan restricts deploys to Singapore during peak hours and has small limits.
- The project in a **private GitHub repository** (`.env` and `storage/` are already in `.gitignore`: never commit them).
- A **new, empty Supabase project for production** (pick the Singapore region to match Railway). Your development database
  holds test users and uploads whose photo files only exist on your computer, so don't reuse it.
  In the new project: Connect, **Session pooler**, copy the string, and use a password you haven't shared anywhere.
- A new `JWT_SECRET`: `python -c "import secrets; print(secrets.token_urlsafe(48))"`

## Steps

1. **Create the project.** Railway dashboard, New Project, Deploy from GitHub repo, pick your repo. Railway finds the `Dockerfile`
   and `railway.json` by itself. The first deploy will **fail until the variables below exist**. That is expected.
2. **Set the region.** Service, Settings, Region: *Southeast Asia (Singapore)*. If you can't attach a volume in that region (Railway's
   docs are inconsistent about this), choose the nearest region that allows it. Changing region later migrates the volume and causes downtime.
3. **Add a volume.** Right-click the project canvas (or Ctrl+K), Volume, attach it to the service, mount path **`/data`**.
   The Dockerfile expects `/data`: photos go to `/data/uploads` and DeepFace's models to `/data/.deepface`.
4. **Add variables** (Service, Variables, Raw Editor):

   ```
   DEEPFACE_POSTGRES_URI=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
   JWT_SECRET=<the 48+ character secret>
   COOKIE_SECURE=true
   ALLOW_SIGNUP=true
   FACE_DETECTOR=opencv
   FACE_WORKERS=1
   FACE_MAX_SIDE=1600
   MAX_UPLOAD_MB=10
   ```

   `FACE_DETECTOR=opencv` is the light option. TensorFlow alone uses about 650 MB, so on a service limited to ~1 GB there is little room for
   models. Switch to `retinaface` (finds more faces, uses more memory) only after you have raised the memory limit.
   `ALLOW_SIGNUP=true` is needed for the Android app's sign-up (set `false` if nobody should self-register).
   URL-encode special characters in the database password. Don't set `PORT`; Railway provides it.
   **Don't copy your local `.env` into Railway.** In particular `UPLOAD_DIR` and `DEEPFACE_HOME` are already set by the Dockerfile to
   the volume (`/data`); a copied `UPLOAD_DIR=storage/uploads` makes photos vanish on every redeploy.
5. **Public URL.** Service, Settings, Networking, Generate Domain.
6. **Deploy.** Saving variables redeploys. Watch Deployments, Logs. It is healthy when `/health` passes (allow a few minutes the first time).
7. **Create the first admin** from your computer, against the production database:

   ```bash
   npm i -g @railway/cli        # once
   railway login
   railway link                 # pick this project and service
   railway run python create_user.py --email you@example.com --name "Your Name" --role admin
   ```

   `railway run` runs the command on your machine with the service's variables, so it needs your local virtualenv from development.

## Check it works

1. `https://<domain>/health` returns `{"status":"ok"}`, and `/docs` loads.
2. Sign in on the web page, add a photographer, upload **one** photo. The first upload is slow: DeepFace downloads its models to the volume once.
3. Search with another photo of that person.
4. Android app: set `EXPO_PUBLIC_API_URL=https://<domain>` in the app's `.env`.

## Memory: raise the service's limit to at least 2 GB

Measured with the real app (CPU, Facenet model, `opencv` detector), peak memory of the app process:

| Situation | Peak memory |
|---|---|
| App started, TensorFlow loaded | about 570 MB |
| Face model loaded (after the first search or upload) | about 850 MB |
| One 12-megapixel photo, **not** shrunk | about 3,800 MB |
| The same photo shrunk to 1600 px (what the app now does) | about 1,370 MB |

The server shrinks photos to `FACE_MAX_SIDE` (default 1600 px, longest side) for face detection only. The stored original and the thumbnail
are untouched. A service limit near 1 GB (for example 953 MB) cannot work: the app is killed the moment the model is loaded and a photo
arrives, and a killed app prints nothing. Set the memory limit in the service's settings (look for the resource or replica limits) to
**2 GB or more**; Railway bills the memory you actually use, not the limit. Raise it further (3 to 4 GB) before raising `FACE_MAX_SIDE` or
`FACE_WORKERS`.

How to see it happening: the log prints `WARNING: memory … of … MB limit` when usage passes 85% of the limit, and after a restart it prints
`PREVIOUS RUN ENDED MID-JOB: search/upload …` naming the job that was running when the app died.

## Running it

- **Memory is the main cost and risk.** Railway bills usage (at the time of writing: memory per GB-month, CPU per vCPU-month, volume per GB).
  Watch the service's Metrics tab for the first days. If it runs out of memory, set `FACE_WORKERS=1` or use `FACE_DETECTOR=opencv`.
- **Redeploys cause a short outage.** A service with a volume can't run two copies at once, so each deploy briefly stops the app
  (Railway documents this). For the same reason you can't add replicas. Moving photos to object storage (S3, Supabase Storage)
  would remove that limit; ask if you want it.
- **Back up both halves together:** the Supabase database and the volume (Railway supports volume backups). Photos and records must stay in sync.
- **Updating:** push to GitHub; Railway rebuilds. Dependency changes rebuild the slow layer, code-only changes are quick.
  If you change `requirements.txt`, regenerate `constraints.txt` (command in its first lines).
- **Database connections:** the app uses at most 5 (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`), plus one at a time inside DeepFace per face job (`FACE_WORKERS`).
  Stay well below your Supabase pooler's limit.
- **One worker only**, by design (login throttling and the face-job limit are per process).

## If something goes wrong

The app prints `[startup]` lines at boot (Python version, memory limit and use, which settings are present, each startup step, and the\ndatabase host with the password hidden). Start by reading those in Deployments, Deploy Logs.

| Symptom | Likely cause |
|---|---|
| Deploy crashes at start with "Set JWT_SECRET" | The variable is missing or shorter than 32 characters |
| "could not translate host name" in logs | `DEEPFACE_POSTGRES_URI` still uses the direct `db.….supabase.co` host. Use the Session pooler string |
| Health check keeps failing | Read the logs. Startup loads TensorFlow, which takes a while; `railway.json` allows 5 minutes |
| App restarts under load, "out of memory" | See Memory above |
| Startup log says `UPLOAD_DIR=… (NOT on a mounted volume …)` | A variable named `UPLOAD_DIR` is set in Railway. Delete it. Entries uploaded before the fix have no files: delete that photographer with their photos (Users tab) and upload again |
| Photos vanish after a redeploy | The volume isn't mounted at `/data`, or the variable `UPLOAD_DIR`/`DEEPFACE_HOME` points elsewhere |
| `free(): invalid pointer` or `Segmentation fault` right after the `[startup] database:` line | A library clash between TensorFlow and a Postgres driver when connecting over SSL. Fixed in this version (`main.py` loads `psycopg2` and `psycopg` first). Keep those imports at the very top |
| `psycopg is an optional dependency` in a search or upload error | DeepFace's database code needs the `psycopg[binary]` package, which it doesn't install itself. It is in `requirements.txt` now. Rebuild, and run `pip install -r requirements.txt` in your local environment too |
| The app restarts during a search or upload, and the logs show nothing before the restart | Killed for using too much memory. After the restart the log says `PREVIOUS RUN ENDED MID-JOB`. See the Memory section: raise the limit to 2 GB or more |
| Logs show the app starting, then restarting every few seconds with no error | Almost certainly **out of memory** (the process is killed silently). Look at the `[startup]` lines: they print the container's memory limit. Raise the memory limit or upgrade the plan (TensorFlow needs about 1 GB just to load, and more once models are loaded) |
| Logs flood with long tracebacks, then "Railway rate limit of 500 logs/sec reached" | A burst of failing requests, usually a batch upload while the database or the app is failing. This version stops a batch after 5 server errors in a row and turns database outages into one short log line. To find the real cause, search the logs for `Error` and read the first exception line, not the frames |
| `MaxClientsInSessionMode` or "max clients reached" | Too many database connections for Supabase's pooler. The app now keeps its pool small (`DB_POOL_SIZE=3`, `DB_MAX_OVERFLOW=2`) and releases connections while DeepFace works. Keep `FACE_WORKERS` low, upload in smaller batches, and avoid running several copies of the app against one database |
| `The database is temporarily unavailable` in the web page | The app couldn't get a database connection (outage, or the limit above). It is safe to retry |
| Build fails installing packages | Send me the build log; I verified the versions resolve on Linux/Python 3.12 but couldn't run Docker here |
