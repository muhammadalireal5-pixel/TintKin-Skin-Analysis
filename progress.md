# Tintkin skin analysis: progress (last updated 2026-10-18, ~01:30)

Goal: CPU-only model that replaces YouCam in the Tintkin web app. Input is a selfie (Cloudinary URL),
output is 0-100 scores (higher = better) for redness, pores, clarity, wrinkles, smoothness, plus skin age.
Served with FastAPI on Hugging Face Spaces.

## Design
- Filters (no training): redness, pores, smoothness, wrinkles, clarity, quality gate. See `skinai/metrics.py`, `skinai/quality.py`.
- Learned model: face age only. MobileNetV3-Large fine-tuned on **FairFace** (CC BY 4.0, 9 age bins),
  age = softmax-weighted bin midpoint, exported to ONNX for CPU. See `scripts/train_age.py`, `skinai/age.py`.
- Skin age = model age shifted by up to about +-8 years from the wrinkle/smoothness/clarity/redness scores (`skinai/scoring.py`).
- Face landmarks: MediaPipe Face Landmarker (Apache 2.0), `models/face_landmarker.task`.

## Done
- FairFace 0.25 config downloaded to `data/fairface/0.25/` (2 train shards + validation). 1.25 config NOT downloaded (stopped; not needed).
- Code written: `skinai/{face,metrics,quality,scoring,age,analyze}.py`, `app.py` (FastAPI), `scripts/{train_age,calibrate}.py`.
- `face.py` ported to the MediaPipe Tasks API (mediapipe 1.1.1 has no `mp.solutions`).
- Added a `pores` metric; fixed a jawline index crash.
- Pipeline tested on FairFace validation images (landmarks, quality gate, metrics, scores all run).
- Deps installed in `.venv` (torch 2.5.1+cu121, mediapipe, timm, onnxruntime, fastapi, etc.). GPU works for training.
- Image bytes packed into `data/fairface_cache/` (memmap) because the machine has only 7.9 GB RAM.
  The first training run crashed with an OOM before this fix.

## Training status
| Epoch | Train loss | Val acc | Val age error |
|---|---|---|---|
| 0 | 1.923 | 48.5% | 6.65 y |
| 1 | 1.441 | 53.2% | **5.93 y** (saved to `models/age_best.pt` at 01:30:45) |

**Training was deliberately stopped at the start of epoch 2 (01:31).** `models/age_best.pt` holds the epoch 1 weights.
`models/age.onnx` has NOT been exported yet, so resuming (step 2 below) is required.

**Update 01:44:** I resumed once (`--init models/age_best.pt --epochs 5 --lr 5e-4`), then stopped it again at iteration ~550 of
epoch 0 of that run, BEFORE it finished an epoch (loss was about 1.32). Nothing from that partial run was saved, so
`models/age_best.pt` is still the epoch 1 weights from 01:30:45 (5.93 y). The resume command below is therefore
still correct and starts from the same point. Log of the first run: `runs/train_age_run1.log`.

- Run started with: `scripts/train_age.py --epochs 8 --batch 128 --workers 3` (about 12 min/epoch).
- Best checkpoint is saved to `models/age_best.pt` each time val error improves. The script does NOT save
  optimizer or schedule state, so it cannot resume exactly.
- `models/age.onnx` is only written at the END of a full run.

## Tomorrow: resume
1. Confirm `models/age_best.pt` exists (epoch 1 weights, 5.93 y).
2. Restart from the saved weights:
   ```
   .venv\Scripts\python.exe -I scripts\train_age.py --init models\age_best.pt --epochs 5 --lr 5e-4 --workers 3
   ```
   (about 1 hour; keep the PC awake; close other heavy apps, as RAM is tight)
3. Recalibrate: `.venv\Scripts\python.exe -I scripts\calibrate.py --n 1000`.
   The current `models/calibration.json` was fit on upsampled 224px FairFace crops, which are NOT representative of real selfies.
   Better: put ~20-50 real selfies in a folder and calibrate on those (needs a small script change).
4. Test end to end: `uvicorn app:app --port 7860`, POST an image to `/analyze/file`.
5. Sanity-check the scores on a few real faces (clear skin vs. visible redness/wrinkles).
6. Write the Dockerfile and requirements.txt for Hugging Face Spaces (CPU, port 7860). Use `onnxruntime`, not torch, in the image.

## Open items and caveats
- Wrinkles, pores, redness, smoothness and clarity are filter heuristics, not validated against any ground truth.
  No commercially licensed wrinkle dataset was found. If the filters are weak, self-label a small set later.
- FairFace age is binned (expect roughly 5-6 y error at the end of training), and "skin age" is a blend, not a measured quantity.
- Quality-gate thresholds (blur/dark/pose) are untested on real selfies.
- `app.py` only accepts `https://res.cloudinary.com/` URLs. Adjust if the Cloudinary cloud uses a custom domain.
- Add auth (API key header) to `app.py` before exposing it publicly.
- Never commit `.env` (has Roboflow and Kaggle keys). `data/`, `runs/`, `*.pt`, `*.onnx` are already gitignored.
- (removed) `scripts/download_data.py` Roboflow downloader was unused and deleted in the 2026-10-19 cleanup.

## Update 2026-10-18 (afternoon): training finished + filter work
- Run 2 finished (`--init age_best.pt --epochs 5 --lr 5e-4`): val age error 5.59 / 5.49 / 5.42 / 5.18 / **5.09 y**, acc 57.9%.
  `models/age.onnx` exported (best 5.09 y). Log: `runs/train_age_run2.log`. Steps 1-2 of "Tomorrow: resume" are DONE.
- Real-photo test (Test2.jpeg, user is 19, has a beard): face age 28.9 y; YouCam said 41. Smoothness 0.0 / pores 2.5 were wrong.
- Root cause found: skin mask counted beard/mustache/hair strands as skin, so texture/pores/dark-spot filters measured hair.
- Fix in `skinai/face.py`: face-skin segmentation (`models/selfie_multiclass_256x256.tflite`, class 3) + `_drop_hair()`
  (removes dark/busy beard-like pixels with a margin). Mask now looks right on Test2. Smoothness 0 -> 15, pores 2.5 -> 31, clarity 58 -> 88.
- Still open: calibration.json is fit on upsampled FairFace crops, so texture/pores still read too harsh on sharp real selfies.
  Need ~20-50 real selfies (varied skin tones, lighting, with/without facial hair) to recalibrate. `calibrate.py` needs a `--dir` option.
- Quality gate "blurry" (Laplacian var < 20) now fires on Test2 because beard edges no longer inflate it. Threshold needs real-selfie data.
- Beard also ages the face for the age model (28.9 vs real 19). Consider beard-masked/blurred crop or augmentation later.

## Update 2026-10-19: context + skin-scan evaluation
- Context: this `model/` project replaces YouCam in the web app at `D:\files for me\Tintkin\TintKin` (YouCam code: `app/lib/youcam.js`, `tests/youcam.test.mjs`, `lib/constants/{credits,metrics}.js`).
- User suggested reusing GitHub repo "scan-skan" by DurtyDhiana instead of building from scratch. Real name: **https://github.com/DurtyDhiana/skin-scan** (MIT, 16 stars, 5 commits, last push 2025-10-27, 153 KB).
- What it is: FastAPI + OpenCV + MediaPipe FaceMesh. `POST /scan` (image) returns 7 maps (redness, oiliness, texture, pores, blemishes, hydration, pigmentation), each scored 0-1 with a base64 RGBA heatmap overlay and regions (forehead/nose/cheeks/chin).
- Findings: it is classical CV heuristics (LAB a*, HSV thresholds, LBP/Laplacian, DoG blobs), NOT a pretrained learned model; no pretrained weights visible (optional blemish CNN, unclear). So it is the same kind of approach as our `skinai/metrics.py` filters. It has NO age / wrinkle / skin-age output.
- Not yet done: clone it and read `src/` to compare against our filters; decide whether to borrow heatmaps/oiliness/pigmentation/hydration. Our hair/beard-aware skin mask (`skinai/face.py`) and the age model are things it lacks.

## Update 2026-10-19 (later): skin-scan vs ours, tested
Cloned to `D:\files for me\Tintkin\skin-scan-eval\repo` (own venv there; needs mediapipe 0.10.14, since ours uses 1.1.1). Harness: `skin-scan-eval/scripts/compare.py`
(runs both on Test2/test3/TestFace with brightness +-30%, blur, grain, red tint).
- skin-scan has NO weights (`src/ml/models` empty). Pure heuristics, every map min-max normalised per face, so scores are relative to that face.
- skin-scan results: `pores` = 0.0 on every image/variant (broken). Oiliness swings 0.26 -> 0.34 / 0.12 with brightness and 0.04 with grain. Red tint moved redness the wrong way on TestFace (0.345 -> 0.302). Blurring TestFace raised texture 0.15 -> 0.23 (wrong way). Hydration is noise-driven.
- Ours: scores stable under brightness/tint (about +-3 pts), blur improves and grain worsens (correct direction). BUT `redness` is saturated at 100 everywhere and TestFace clamps to 100 on most metrics = calibration is the weak point, not the design.
- Decision (recommended): keep ours as the scoring engine. From skin-scan borrow only the heatmap overlay idea (`visualize.py`) if TintKin wants maps.
- Next: recalibrate on real selfies; ideally with YouCam scores as reference for ~30 selfies (pull from TintKin DB/exports) and fit our scores to them. Fix redness saturation.
- Caveat: no ground truth used; this only measures consistency, not accuracy.

## Update 2026-10-19 (pair test: Test2 vs test3, same person before/after shower)
Memory rule: never suggest YouCam data/DB again (see memory `feedback-no-youcam-data`).
Ours (Test2 -> test3): wrinkles 76.0->76.8, smoothness 89.1->90.8, firmness 80.6->84.4, pores 90.9->96.2, clarity 96.2->85.9, radiance 97.6->85.8, spots 90.6->64.8, overall 86.2->78.0, redness 100->100 (saturated). Face age 18.9->21.0 (real 19).
skin-scan (0-1): redness .545->.517, oiliness .257->.285, texture .119->.132, pores 0->0 (dead), blemish .169->.122, hydration .092->.008, pigment .193->.207.
Findings: ours stable on wrinkles/smoothness/firmness/pores, but spots (-26), radiance (-12), clarity (-10) swing; raw dark_spots .163->.231, radiance_unevenness .71->.84, lighting_imbalance .10->.19, tone_L .35->.42, sharpness 32->18. Lighting differs between the two photos (side window light in test3), so illumination flattening is not fully removing it.
skin-scan only looks consistent because its outputs sit in a narrow band (pores 0, texture ~.12, pigment ~.2); hydration still moved 11x. gray-world also uses whole frame, so shirt colour (grey vs blue) shifts skin colour.
Next: make spots/radiance/clarity lighting-robust (stronger local flat-field / higher-percentile based, test on these two + perturbations), fix redness saturation.

## Update 2026-10-19 (pipeline 0.3.0): lighting/edge robustness pass
Diagnosis (scratch scripts, not in repo): the old `dark_spots` / `tone_*` were dominated by the skin-mask BORDER (hairline, beard line, eyelids, nostrils, lips) and by under-eye shading, not pigmentation. Cheeks were identical between Test2 and test3.
Changes in `skinai/metrics.py`: new `_core()` (skin minus under-eye, nose, border eroded `CORE_ERODE=15`) and `EVAL_ERODE=11` extra px inside it. `dark_spots`, `tone_L`, `tone_chroma` (so spots, radiance, clarity) now use it; blurs use the core mask only.
Recalibrated 4 keys on the 16 real photos in `test/` (`models/calibration.json`, others unchanged). `PIPELINE_VERSION` = 0.3.0. Backups: `skinai/metrics.py.v0.2.0.bak`, `models/calibration.v0.2.0.json`.
Test2 -> test3 gap (points): spots 26 -> 11, radiance 12 -> 3, clarity 10 -> 4, overall 8 -> 2. Wrinkles/firmness/smoothness/pores unchanged (<=5). Exposure +-30% and red tint move spots/radiance/clarity by <=3. Blur improves and grain worsens scores (correct direction).
Still open: spots gap 11 (> 5 target; raw 0.025 vs 0.067, tiny absolute values but steep score slope, Test2 clamps to 100). Redness still 100 on both: raw is ~0 because the measure is relative to the person's own median, so this skin genuinely has little a* excess; not changed. Only 16 calibration photos. Scores for a different sharpness/resolution than FairFace or these photos are untested.

## Update 2026-10-19: cleanup + decision
- Decision: we are NOT using DurtyDhiana/skin-scan (it was tested, see above). `skin-scan-eval/` (clone + venv, ~840 MB) deleted.
- Deleted: all `*.bak*` files, `app.py.bak`, `scripts/download_data.py`, superseded calibration files (kept only `models/calibration.json`), duplicate run-2 age weights (identical to `age.onnx` / `age_best.pt`), `__pycache__`.
- Kept: `models/age_run4_5.08.onnx` + `age_best_run4_5.08.pt` (alternative age model, 5.08 y vs 5.09 y in service), `data/` (FairFace, needed to retrain/recalibrate), `runs/` logs, `test/` (16 calibration photos), `flow.md`.
- Added `scripts/consistency_check.py` (brightness/tint/blur/grain stability check, from the comparison harness).
- Pipeline re-run after cleanup: identical scores on Test2.

## Update 2026-10-19: API tested end to end (uvicorn, localhost:7860)
Tested with `SKINAI_API_KEY` set. PASS: `/health`; 401 with no/wrong `x-api-key`; 400 for non-Cloudinary URL, `https://res.cloudinary.com@evil.com/...`, and `http://`; 400 for missing Cloudinary image (404 upstream) and for non-image upload; `/analyze/file` (Test2, 200, ~2.4 s); `/analyze/url` and `/v1/skin-analysis` with real Cloudinary URLs from the public `demo` cloud (plain, and a TintKin-style `c_thumb,g_face` crop, with masks: 4 PNG masks ~14-33 KB each, ~89 KB body, ~4.5 s). No-face image -> 200 `ok:false, issues:[no_face]`.
Fixes to `app.py`: URL check now parses the URL (https, host exactly `res.cloudinary.com`, no userinfo/port, and path `/<CLOUDINARY_CLOUD_NAME>/image/` when that env var is set, same rule as TintKin `lib/utils/cloudinary.js`); body streamed with a 10 MB cap; network errors -> 400; analysis moved to a worker thread behind a 1-slot semaphore (it blocked the event loop before; models are shared singletons so scans run one at a time). `/health` now answers in ms during scans.
Added `README.md` (HF Space config: sdk docker, app_port 7860). `requirements.txt` matches the tested venv.
NOT tested: Docker build / HF Space itself (no Docker here). Throughput: ~2.2 s per scan, one at a time, so ~0.45 scans/s per Space on this CPU (HF CPU may be slower). First request after Space sleep is a cold start.
To deploy: set Space secrets `SKINAI_API_KEY` and `CLOUDINARY_CLOUD_NAME`; `.gitignore` ignores `*.onnx` so push `models/age.onnx` via Git LFS (or un-ignore it); point TintKin at `POST <space>/v1/skin-analysis` with the `x-api-key` header.

## Update 2026-10-19: integration guide + pushed to GitHub
- `INTEGRATION_GUIDE.md` written (HF hosting, API reference, TintKin wiring via new `app/lib/skinai.js` + one-line swap at `actions.js:516`, score caveats). Guide is documentation only: TintKin code NOT modified yet.
- Important: this replaces YouCam `analyzeSkin` only. YouCam `simulateSkin` (what-if) is still needed.
- Repo pushed: https://github.com/muhammadalireal5-pixel/TintKin-Skin-Analysis (branch main, commit 764ec6d). Models via Git LFS (age.onnx, face_landmarker.task, selfie_multiclass tflite). Excluded: `.env`, `data/`, `test/` + Test*.jpg photos, other weights, `.venv`.
- Not done: HF Space creation/push (needs user's HF account/token), Docker build test, TintKin code change.
