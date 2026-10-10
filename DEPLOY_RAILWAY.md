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
   FACE_DETECTOR=retinaface
   FACE_WORKERS=2
   MAX_UPLOAD_MB=10
   ```

   `ALLOW_SIGNUP=true` is needed for the Android app's sign-up (set `false` if nobody should self-register).
   URL-encode special characters in the database password. Don't set `PORT`; Railway provides it.
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

## Running it

- **Memory is the main cost and risk.** Railway bills usage (at the time of writing: memory per GB-month, CPU per vCPU-month, volume per GB).
  Watch the service's Metrics tab for the first days. If it runs out of memory, set `FACE_WORKERS=1` or use `FACE_DETECTOR=opencv`.
- **Redeploys cause a short outage.** A service with a volume can't run two copies at once, so each deploy briefly stops the app
  (Railway documents this). For the same reason you can't add replicas. Moving photos to object storage (S3, Supabase Storage)
  would remove that limit; ask if you want it.
- **Back up both halves together:** the Supabase database and the volume (Railway supports volume backups). Photos and records must stay in sync.
- **Updating:** push to GitHub; Railway rebuilds. Dependency changes rebuild the slow layer, code-only changes are quick.
  If you change `requirements.txt`, regenerate `constraints.txt` (command in its first lines).
- **One worker only**, by design (login throttling and the face-job limit are per process).

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| Deploy crashes at start with "Set JWT_SECRET" | The variable is missing or shorter than 32 characters |
| "could not translate host name" in logs | `DEEPFACE_POSTGRES_URI` still uses the direct `db.….supabase.co` host. Use the Session pooler string |
| Health check keeps failing | Read the logs. Startup loads TensorFlow, which takes a while; `railway.json` allows 5 minutes |
| App restarts under load, "out of memory" | See Memory above |
| Photos vanish after a redeploy | The volume isn't mounted at `/data`, or the variable `UPLOAD_DIR`/`DEEPFACE_HOME` points elsewhere |
| Build fails installing packages | Send me the build log; I verified the versions resolve on Linux/Python 3.12 but couldn't run Docker here |
