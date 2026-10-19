# TintKin Skin Analysis: integration guide

This service replaces **YouCam skin analysis** (`analyzeSkin` in `TintKin/app/lib/youcam.js`) with our own model.
It does **not** replace YouCam skin *simulation* (`simulateSkin`, the what-if sliders). Keep `YOUCAM_API_KEY` for that.

```
TintKin (Next.js, Vercel)                         Hugging Face Space (this repo, Docker, CPU)
  analyzeSkinOwn(imageUrl)  --POST /v1/skin-analysis-->  fetch image from Cloudinary
  { image_url, real_age }      x-api-key header          face + skin segmentation -> filters -> 0-100 scores
                            <---- { output: [...] } ---  + face-age model (ONNX) -> skin age
```

The reply is shaped like YouCam's (`output[]` with `wrinkle`, `firmness`, `age_spot`, `radiance`, `all`, `skin_age`),
so TintKin's existing `extractScoreInfo` / `readScore` code in `actions.js` reads it unchanged.

---

## 1. Hosting on Hugging Face

Hugging Face does not run a GitHub repo directly. A Space is its own git repo. Use one of these:

**A. Push to the Space repo (simplest)**
1. huggingface.co -> New Space -> SDK **Docker**, hardware **CPU basic** (free), name e.g. `tintkin-skin-analysis`.
2. Add the Space as a second remote and push:
   ```
   git remote add space https://huggingface.co/spaces/<hf-username>/tintkin-skin-analysis
   git push space main
   ```
   Log in with a Hugging Face **write** token as the password. `.gitattributes` already tracks the model files
   (`*.onnx`, `*.tflite`, `*.task`) with Git LFS, which Hugging Face requires for files over 10 MB, so install `git lfs` first.

**B. Auto-sync from GitHub** (`.github/workflows/sync-to-hf.yml`, needs a repo secret `HF_TOKEN`)
```yaml
name: Sync to Hugging Face
on: { push: { branches: [main] }, workflow_dispatch: {} }
jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0, lfs: true }
      - run: git push --force https://<hf-username>:${{ secrets.HF_TOKEN }}@huggingface.co/spaces/<hf-username>/tintkin-skin-analysis main
```

**Space settings -> Variables and secrets** (secrets, not variables):

| Name | Value |
|---|---|
| `SKINAI_API_KEY` | a long random string. Every analyze call must send it as `x-api-key`. If unset, the API is open. |
| `CLOUDINARY_CLOUD_NAME` | same value as TintKin's `NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME`. Restricts `image_url` to that cloud. |

The Space URL is `https://<hf-username>-tintkin-skin-analysis.hf.space`. Keep the Space **public** and rely on
`x-api-key`. A private Space also requires an `Authorization: Bearer hf_...` header on every request.

Check it: `curl https://<space-url>/health` -> `{"status":"ok"}`.

Notes:
- Free Spaces sleep after about 48 h idle; the first request then waits for a cold start (up to a minute or two).
  Ping `/health` on a schedule if that matters, or use paid hardware.
- The Docker image has not been built on the dev machine (no Docker there). The app itself was tested with uvicorn.
  Watch the first build log on the Space.

---

## 2. API

Base URL: the Space URL. All analyze calls need `x-api-key: <SKINAI_API_KEY>` (401 otherwise).

### `POST /v1/skin-analysis` (use this one)
```json
{ "image_url": "https://res.cloudinary.com/<cloud>/image/upload/<id>.jpg", "real_age": 24, "masks": false }
```
- `image_url`: required. `https`, host `res.cloudinary.com`, path `/<CLOUDINARY_CLOUD_NAME>/image/...`
  (`upload`, `authenticated` and signed URLs all work). Redirects are not followed. Max 10 MB.
- `real_age`: optional, 5-110. Anchors the skin-age estimate (70% real age, 30% model age, clamped to +-10 years).
- `masks`: default `true`. Adds an 8-bit PNG per metric (`mask_png_b64`), about 15-35 KB each. Send `false` unless you use them.

Response (200):
```json
{
  "ok": true, "issues": [], "confidence": 1.0, "comparable": true,
  "quality": { "face_fraction": 0.23, "sharpness": 32.0, "brightness": 145.4, "...": "..." },
  "output": [
    { "type": "wrinkle",  "ui_score": 76.0, "raw_score": 0.0132 },
    { "type": "firmness", "ui_score": 80.6, "raw_score": 0.0092 },
    { "type": "age_spot", "ui_score": 100.0, "raw_score": 0.0252 },
    { "type": "radiance", "ui_score": 100.0, "raw_score": 0.5819 },
    { "type": "all",      "score": 89.1 },
    { "type": "skin_age", "score": 15.0 }
  ],
  "face_age": 18.9,
  "extras": { "smoothness": 89.1, "pores": 90.9, "redness": 100.0, "clarity": 100.0 },
  "pipeline_version": "0.3.0", "calibration_version": "b2fd6ea4"
}
```
Scores are 0-100, **higher is better**. `output` is `[]` when no face was found.

### `ok`, `issues`, `confidence`, `comparable`
`ok: false` means a photo-quality problem; scores (if present) are less reliable. `issues` may contain:

| issue | meaning | tell the user |
|---|---|---|
| `no_face` | no face detected (`output` is empty) | "No face found. Face the camera, hair and hands off your face." |
| `not_enough_skin_visible` | too little visible skin (`output` empty) | same as above |
| `face_too_small` | face under 256 px wide | "Move closer." |
| `blurry` | out of focus | "Hold still / tap to focus." |
| `too_dark` / `overexposed` | exposure | "Use even daylight." |
| `uneven_lighting` | strong side light | "Face a window, not beside it." |
| `face_not_frontal` / `head_tilted` | pose | "Look straight at the camera." |
| `skin_smoothed` | beauty filter or heavy upscaling | "Turn off beauty filters." |
| `multiple_faces` | more than one face | "Only one person in frame." |

`confidence` is 1.0 minus a penalty per issue. `comparable: false` (confidence below 0.7) means **do not compare this scan
with earlier ones** (progress charts, streaks, deltas); store it but flag it.

### Other endpoints
- `POST /analyze/url`: same body, returns the native result (`scores`, `raw` measures, `face_age`, `skin_age`) instead of `output[]`.
- `POST /analyze/file`: multipart `file` (+ optional `real_age`). For testing with local images.
- `GET /health`.

### Errors
| HTTP | cause |
|---|---|
| 400 | `image_url` is not an allowed Cloudinary URL, image could not be fetched (e.g. 404 upstream), or not a decodable image |
| 401 | missing/wrong `x-api-key` |
| 413 | image over 10 MB |
| 422 | malformed request body |

---

## 3. Wiring it into TintKin

### 3.1 Environment (Vercel + `.env.local` + `.env.example`)
```
SKINAI_URL=https://<hf-username>-tintkin-skin-analysis.hf.space
SKINAI_API_KEY=<same value as the Space secret>
```

### 3.2 New file `app/lib/skinai.js`
```js
const URL_BASE = process.env.SKINAI_URL;
const KEY = process.env.SKINAI_API_KEY;

const ISSUE_MESSAGES = {
  no_face: "No face was detected. Face the camera with your hair and hands clear of your face.",
  not_enough_skin_visible: "We couldn't see enough of your face. Face the camera in good light.",
  multiple_faces: "Please make sure only one person is in the photo.",
};

export async function analyzeSkinOwn(imageUrl, realAge) {
  const res = await fetch(`${URL_BASE}/v1/skin-analysis`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": KEY },
    body: JSON.stringify({ image_url: imageUrl, real_age: realAge ?? undefined, masks: false }),
    signal: AbortSignal.timeout(45000), // cold start + queue; a scan itself takes ~2-5 s
  });
  if (!res.ok) throw new Error(`Skin analysis failed (HTTP ${res.status})`);
  const data = await res.json();
  if (!data.output?.length) {
    const issue = data.issues?.find((i) => ISSUE_MESSAGES[i]);
    const err = new Error(ISSUE_MESSAGES[issue] || "We couldn't analyze this photo. Please try another.");
    err.isUserFacing = true;
    throw err;
  }
  return data; // { output: [...], ok, issues, comparable, confidence, pipeline_version, calibration_version }
}
```

### 3.3 Swap the call in `app/lib/actions.js` (line ~516)
```js
// before
const youCamResult = await analyzeSkin(signCloudinaryUrl(imageUrl));
// after
const youCamResult = await analyzeSkinOwn(signCloudinaryUrl(imageUrl), user.age);
```
Everything below it (`youCamResult.results || ... || youCamResult`, `extractScoreInfo`, `readScore`) works as is,
because `data.output` is an array. Add `import { analyzeSkinOwn } from "./skinai";`. Leave the `youcam` import for `simulateSkin`.
`user.age` is a placeholder: pass the real age if TintKin stores one, else omit it.

### 3.4 Recommended follow-ups
- **Store versions** on each `Selfie`: `pipeline_version`, `calibration_version`, `comparable`, `confidence`, `issues`.
  Never compare scores across different versions, and skip non-`comparable` scans in trend charts.
- **Cost constant:** `YOUCAM_COST_PER_SCAN` (0.432) in `actions.js` no longer applies to analysis scans.
- **Retries:** TintKin's 4-candidate crop retry (`applyFaceCropToCloudinary`) isn't needed for this API. For `face_too_small`
  you can retry once with a crop that keeps the width at or above about 1000 px. Avoid crops that upscale a small image:
  that triggers `skin_smoothed`.
- **Masks:** `masks: true` returns base64 PNGs, not URLs (a stateless service can't host them). If you want overlays,
  upload each `mask_png_b64` to Cloudinary and store the URL as `mask_urls`.
- **Timeouts/queueing:** one scan runs at a time per Space (the face models are shared). Expect about 2-5 s each on CPU, so a
  burst of N users waits N x ~2.5 s. If you outgrow that, run more Spaces or move to bigger hardware.

---

## 4. What to expect from the scores

- Scores are calibrated on a small reference set (16 real photos). They are **relative, not clinical**.
- Stability: on one person's two photos taken close together, wrinkles, smoothness and radiance differed by under 3 points,
  firmness, clarity and pores by 4-5, and spots by about 11. Treat changes smaller than about 5 points as "no change" in the UI.
- `spots`, `radiance` and `clarity` read the cheek/forehead interior only (they ignore the hairline, beard line, eye and nose areas).
- `redness` is measured against the person's own median skin tone, so it reads 100 when there is little excess redness.
- Wrinkle/firmness/spots/radiance are filter measurements, not a model trained on dermatology labels. Only face age is a trained model.

## 5. Troubleshooting

| Symptom | Check |
|---|---|
| 401 | `x-api-key` header equals the Space secret `SKINAI_API_KEY` |
| 400 "must be a Cloudinary URL" | host must be exactly `res.cloudinary.com`, path must start with `/<CLOUDINARY_CLOUD_NAME>/image/` |
| 400 "could not fetch image (401/404)" | asset is `authenticated` and unsigned, or the public id is wrong; sign it with `signCloudinaryUrl` |
| First request very slow or times out | Space was asleep; retry, or ping `/health` |
| Space build fails on model files | `git lfs` not installed when pushing; re-push the LFS files |
| `ok: false` with `skin_smoothed` | upscaled or beauty-filtered photo; send the original |
