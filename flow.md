# flow.md: change log and data flow (kept in order, newest at the bottom)

Rule: no changes to project code without the user's approval. Entries below say what was added, where,
which functions it uses, and what each function hands to which file/function.

---

## 0. Existing pipeline (read-only reference, no changes made)

Entry: `app.py` (FastAPI) -> `skinai/analyze.py::analyze(data: bytes)`

| # | Call (file::function) | Input | Output | Goes to |
|---|---|---|---|---|
| 1 | `face.py::load_image` | image bytes | BGR array, longest side <= 1024 | `detect_face` |
| 2 | `face.py::detect_face` | BGR array | `Face` (landmarks, bbox, n_faces, seg) or None | `check_quality`, `compute_raw` |
| 2a | `face.py::_get_landmarker` | -- | MediaPipe FaceLandmarker (cached) | `detect_face` |
| 2b | `face.py::face_skin_segmentation` | BGR array | uint8 mask, 255 = face skin (class 3) | stored in `Face.seg` |
| 3 | `quality.py::check_quality` | `Face` | `QualityReport` (ok, issues, stats) | `analyze` (gate) |
| 3a | `Face.skin_mask_cached` -> `Face.skin_mask` | `Face` | final skin mask (cached on the Face) | `check_quality`, `compute_raw`, `Face.region` |
| 3b | `Face.poly_mask(idx)` | landmark index list | filled convex-hull mask | `skin_mask`, `region`, zone masks in metrics |
| 3c | `Face._drop_hair` | mask | mask minus dark/busy (beard-like) pixels | end of `skin_mask` |
| 4 | `metrics.py::compute_raw` | `Face` | dict of raw measures | `scoring.score_raw` |
| 4a | `Face.normalized(512)` | `Face` | Face rescaled to 512px face width | `compute_raw` |
| 4b | `_lab`, `_masked_blur`, `_ridge` | arrays | Lab image, masked blur, Frangi-like ridge map | `compute_raw` |
| 5 | `scoring.py::score_raw` | raw dict | 0-100 scores (redness, smoothness, wrinkles, pores, clarity) | `analyze` |
| 5a | `scoring.py::load_calibration` / `to_score` | `models/calibration.json` | p5/p95 ranges, linear map | `score_raw` |
| 6 | `age.py::predict_age` (+ `Face.crop_face`) | 224px face crop | face age (years) via `models/age.onnx` | `scoring.skin_age` |
| 7 | `scoring.py::skin_age` | face age + scores | skin age | `analyze` result |

Skin-mask stages inside `Face.skin_mask` (in order):
1. face oval hull minus dilated eyes/brows/lips
2. erosion (`0.015 * image width`, fixed, not tied to face size)
3. YCrCb colour prior (fallback to geometric mask if it removes > 50%)
4. AND with segmenter mask (skipped if it removes > 50%)
5. `_drop_hair`

---

## 1. 2026-10-18: Step 1 diagnostic (read-only, nothing in `skinai/` changed)

Purpose: see what is wrong with the pipeline on real photos before changing anything.
Inputs: `test/*.jpg` (16 images), `Test2.jpeg`, `TestFace.png`.
CPU only (`CUDA_VISIBLE_DEVICES=` empty) so the age training on the GPU is not disturbed.

### 1a. `step1_diag.py`
Path: `C:\Users\FBC\AppData\Local\Temp\claude\D--files-for-me-Tintkin-model\57fc284c-7647-467d-a0c4-d65fb482e893\scratchpad\step1_diag.py`
What it does: runs the real pipeline functions per image and records the numbers; draws a 3-panel overlay
(original | segmenter mask | final skin mask, with the face bbox).

Functions used and what they give it:
- `face.load_image(path)` -> image (<= 1024px)
- `face.detect_face(img)` -> `Face`
- `quality.check_quality(face)` -> issues + stats (face_fraction, sharpness, brightness, lighting_imbalance, yaw, roll)
- `Face.skin_mask_cached()` -> final mask (for the overlay and mask_vs_oval ratio)
- `Face.poly_mask(F.FACE_OVAL)` -> oval area, used as the denominator for `mask_vs_oval`
- `metrics.compute_raw(face)` -> raw measures
- `scoring.score_raw(raw)` -> 0-100 scores
- `age.predict_age(face.crop_face(224, 0.15))` and `scoring.skin_age` -> face/skin age

Outputs (all in `...\scratchpad\step1\`): `results.json` and `<image>_overlay.jpg` per image.

### 1b. `stages.py`
Path: `...\scratchpad\stages.py`
What it does: repeats the steps of `Face.skin_mask` one by one (same code as in `face.py`, copied, not imported)
and prints how much of the face oval survives each stage. It uses `load_image`, `detect_face`, `poly_mask`,
`Face.seg` and `Face._drop_hair`.

### 1c. Findings (numbers are from the runs above)
1. **`_drop_hair` is the biggest mask killer.** It removes about 30-40% of what is left, even on clean-shaven
   faces. Typical ratio of mask to face oval: geometric 0.5 -> after colour prior 0.5 -> after segmenter 0.5 ->
   after `_drop_hair` 0.2-0.4. It cuts whole cheeks and chins (shadow side, darker skin) in test1/test13/test2.
   The YCrCb prior and the segmenter remove almost nothing (fallback never fired).
2. **`face_too_small` fires on 15 of 17 images.** `face_frac` is bbox area / image area with threshold 0.12; studio
   portraits sit at 0.04-0.1. It measures framing, not resolution, and is not a useful check for these inputs.
3. **Sharpness is not comparable across images.** Laplacian variance ranges 12-800 even after the 1024px cap, so
   the `blurry` threshold (20) mostly reflects image content and processing (retouched skin scores 20-25 and
   fails; test3/test13 with sharp texture score 500-800).
4. **Scores are very bimodal.** smoothness is 0 on 8 of 16 images and pores is 0 on 4, while retouched/smooth
   images (test2, test7, test8, TestFace.png) score 84-100. The calibration range (texture 0.16-0.68, pores 0-0.38)
   was fit on upsampled 224px FairFace crops, and real high-res photos exceed it, so the scores saturate.
5. **Resolution is thrown away twice.** Originals are 2000-6000px; `load_image` shrinks to 1024, then
   `Face.normalized` rescales the face to 512px width. In the original photos the faces are 550-1400px wide, so after
   the 1024px cap (a 4-6x shrink) they are about 100-250px wide and are then upscaled 2-3x to 512 (`norm_scale` 1.3-4.9
   in the run). Pore and texture measures are therefore taken on interpolated pixels, not real detail.
6. **Wrinkles score 100 on 6 images** (raw wrinkle < 0.005), only because the facial zones are small after masking.
7. Face age is 19-33 for these faces (plausible); skin age follows within a few years.

### 1d. Open items for the user's decision (no code changed yet)
- Fix or replace `_drop_hair` (apply it only where there is facial hair, e.g. lower face, or compare against the
  local skin, not the global median).
- Decide the working resolution for pores/texture.
- Replace/rebuild quality thresholds (`face_too_small`, `blurry`).
- Then recalibrate on a larger licence-clean set of high-res faces.

---

## 2. 2026-10-18: Step 2, approved code changes (backups: `skinai/*.bak*`, `scripts/calibrate.py.bak`)

Verified by re-running `step1_diag.py` on the same 18 images after each change (outputs in scratchpad `step2\`, `step3\`).

### 2a. `skinai/face.py`
- NEW `Face.face_width` (property): bbox width in px. Used by `skin_mask`, `_drop_hair`, `quality.check_quality`, `metrics.compute_raw`.
- CHANGED `Face.skin_mask`: eye/brow/lip dilation and edge erosion now scale with `face_width`
  (`0.02*fw` and `0.03*fw`, were 9px fixed and `0.015*image width`). Output unchanged type (uint8 mask) -> `_drop_hair`.
- REWRITTEN `Face._drop_hair(mask)` -> mask:
  - below the nose base (landmark 2 row) = beard zone: drop pixels that are dark (<0.72 x median) AND textured
    (busy > 1.5 x median), or very dark (<0.5 x median), or very busy and dark-ish (old stubble rule). Smooth shadows are kept.
  - above it: drop only pixels < 0.6 x the local skin level (masked Gaussian, sigma 0.12*fw), i.e. hairline strands.
  - result: mask / face oval went from 0.20-0.45 to 0.54-0.68; Test2.jpeg's beard and moustache are still excluded.
- CHANGED `load_image` default `max_side` 1024 -> 2048 (feeds `detect_face`, `compute_raw`).

### 2b. `skinai/metrics.py::compute_raw`
- Texture and pores now use `fd = face.normalized(dw)` with `dw = clip(face_width, 512, 1024)` (native detail, no upsampling beyond the source).
- Sigmas, opening kernel, blob areas and the pore density denominator are scaled by `ks = dw/512` so they cover the same physical size.
- All other measures still use the 512px face `f`.
- Effect: raw texture/pores values are about 1.5-3x higher than before, so the old `calibration.json` saturates smoothness/pores at 0. Needs recalibration.

### 2c. `skinai/quality.py::check_quality`
- `face_too_small`: now `face.face_width < 256` px (was face area / image area < 0.12, which fired on 15 of 17 images).
- Sharpness: Laplacian variance of the whole face box at 512px width (was skin-mask pixels only); `blurry` threshold 20 -> 10.
  Measured: a real blur (sigma 1.5) gives 3-9, sharp or retouched faces 15-600. `stats["sharpness"]` now means this new value.
- Result on the 18 images: only `TestFace.png` (105px wide, upscaled) is flagged too small/blurry; remaining flags are pose and lighting.

### 2d. `scripts/calibrate.py`
- NEW options `--dir <folder>` (real photos via `face.load_image`) and `--out <json>`; default behaviour (FairFace, `models/calibration.json`) unchanged.
- Ran `--dir test --out models/calibration_preview.json` (16 faces; PREVIEW ONLY, not applied). `models/calibration.json` is untouched.

### 2e. Still open
- Apply a calibration fit on a larger real-photo set (16 images is too few; p5/p95 are basically min/max).
- Blown highlights (test9) and specular areas still sit inside the mask.
- Age training run 3 in progress (`runs/train_age_run3.log`).

---

## 3. 2026-10-18: Stopgap calibration applied (user approved "B")
- `models/calibration.json` replaced by the 16-face real-photo fit (`scripts/calibrate.py --dir test`). Read by `scoring.load_calibration` -> `score_raw`.
- Old FairFace calibration saved as `models/calibration_fairface_old.json`; the fit is also kept as `models/calibration_preview.json`.
- PROVISIONAL: 16 stock photos only (p5/p95 ~ min/max). Refit with ~100 varied high-res faces: `python scripts/calibrate.py --dir <folder>`.

---

## 4. 2026-10-18: Age training run 3 (`--init age_best.pt --epochs 5 --lr 3e-4`), log `runs/train_age_run3.log`
- Val age error per epoch: 5.23, 5.48, 5.19, 5.23, 5.21 y (run 2 best was 5.09 y). Train loss fell 0.99 -> 0.75 while val stayed flat, i.e. overfitting.
- Run 3 did NOT beat run 2. Restored `models/age_best.pt` and `models/age.onnx` from `models/age_best_run2_5.09.pt` / `models/age_run2_5.09.onnx` (5.09 y).
- Used by `skinai/age.py::predict_age`.

---

## 5. 2026-10-18: Beard masking for the age model + FairFace 1.25 training data (user approved)

### 5a. Finding that triggered it
FairFace 0.25 train = 86,744 images = the two shards we already had, i.e. the complete set. Nothing "rest of it" exists there.
The 1.25 config holds the same faces at 448x448 with wider padding (2 of 4 train shards were downloaded; the other 2 are being fetched to `data/fairface/1.25/`, log `runs/fairface_125_dl.log`).

### 5b. Beard test (scratchpad `beard_age.py`, read-only) on the 17 photos
Face age with the beard region painted over vs. original: Test2.jpeg (user, 19, beard) **29.0 -> 18.9**; clean-shaven faces mostly within +-2 y
(test15 25.7->19.7, test3 21.8->18.0, test4 21.5->17.8 moved more; likely stubble). Inpaint (Telea) was kept over flat colour fill.

### 5c. Code changes
- `skinai/face.py`: NEW `Face.beard_free_image()` -> image copy with the region (face oval below nose base) minus (lips, clean skin mask) inpainted.
  Uses `Face.skin_mask_cached`, `Face.poly_mask`, `cv2.inpaint`.
- `skinai/face.py`: `Face.crop_face(size, margin, beard_free=False)` new flag; when True it crops from `beard_free_image()`.
- `skinai/analyze.py`: age call is now `A.predict_age(face.crop_face(224, margin=0.15, beard_free=True))`. `predict_age` -> `scoring.skin_age`.
- Result on Test2.jpeg: face age 18.9, skin age 12.8 (true 19). Skin age is now low because `scoring.skin_age` subtracts up to 8 y when the
  provisional calibration gives high scores. Revisit after the real recalibration.
- `scripts/train_age.py`: NEW `--extra125` flag. Builds `data/fairface_cache/train_125.{bin,npz}` from the 1.25 parquet shards (`_build_cache(split, version)`),
  centre-crops each 448px image to the 0.25-equivalent tight view (`TIGHT_FRAC = 1.25/2.25`), and concatenates it with the 0.25 train set.
  Validation stays the 0.25 validation set so results are comparable with runs 2-3 (best 5.09 y).
- Backups: `skinai/analyze.py.bak`, `scripts/train_age.py.bak`.

---

## 6. 2026-10-18: Age training run 4 (`--init age_best.pt --epochs 4 --lr 2e-4 --extra125`), log `runs/train_age_run4.log`
- Data: FairFace 0.25 train + 1.25 train (448px, centre-cropped to the tight view) = 173,488 images. Val = 0.25 validation (10,954).
- Val age error: 5.08 (epoch 0), 5.10, 5.09, 5.12 y. Train loss 1.06 -> 0.82. Overfitting again; val flat vs run 2 (5.09).
- Compared run 2 vs run 4 on the 17 test photos with beard-free crops (scratchpad `cmp_age.py`): run 4 reads older on average (+1.2 y).
  Test2.jpeg (true age 19): run 2 = 18.9, run 4 = 23.6. test4: 17.8 vs 23.3.
- Decision: kept run 2 in service. `models/age.onnx` / `age_best.pt` = run 2 (5.09 y). Run 4 saved as `models/age_run4_5.08.onnx`, `models/age_best_run4_5.08.pt`.
- Conclusion: FairFace's 10-year bins cap this model at about 5 y. Further gains need exact-age data (own consented user data, or a commercially licensed set).

---

## 7. 2026-10-18: Read-only review of the TintKin web app + stability test (no code changed in either project)

### 7a. How TintKin uses YouCam (files read in `D:\files for me\Tintkin\TintKin`)
- `app/lib/youcam.js::analyzeSkin(imageUrl)` -> POST `skin-analysis` with `dst_actions = [wrinkle, firmness, age_spot, radiance]` (`lib/constants/metrics.js`), polls the task, retries with 3 Cloudinary face-crop variants. `extractScoreInfo` normalises the reply to `{wrinkle, firmness, age_spot, radiance: {ui_score, raw_score, mask_urls}, all: {score}, skin_age}`.
- `app/lib/actions.js` (scan flow, ~line 516-700): maps to `scores = {wrinkles, firmness, spots, radiance}`, `overallScore = all.score`, `skinAge = skin_age`; stores a `Selfie` (scores, overallScore, skinAge, `maskUrls`, `youCamTaskId`); caches Qwen advice when all scores are identical to the last scan; streak and badge logic; badge `youth_catalyst` when `skinAge < realAge` (`user.birthDate` exists).
- UI: `skin-summary.js` bands (>=80 Excellent, >=65 Good, >=50 Fair), per-metric delta vs previous scan, weakest metric = focus, "Skin age X, N years younger than you"; `ShareCard`, progress page, weekly average.
- `app/lib/predict.js::projectTrajectory`: per-metric velocity from the user's own scan history (linear regression, blended with baselines), lifestyle/product multipliers.
- What-If: `simulateSkin` -> YouCam `skin-simulation` (generative image), not a scoring call. Our model cannot replace this part.
- YouCam error messages to users: face too small / not found / out of bounds / too dark.

### 7b. Stability test (scratchpad `stability.py`, results `stability_out.txt`)
17 photos x 8 variants (original, brightness +-15%, warm/cool white balance, JPEG q40, noise, 3% crop shift), full `analyze()` with run-2 age model.
Mean std per photo / mean range (max-min): redness 15.4 / 49.6, smoothness 4.2 / 14.6, wrinkles 4.7 / 16.6, pores 6.0 / 19.9, clarity 6.7 / 22.6,
**skin age 1.15 / 3.9 y (worst 7.0)**, face age 0.64 / 2.05 y (worst 5.0).
Conclusion: face age is stable; redness is unusable; the other scores move several points from lighting/compression alone, comparable to real day-to-day change a journal wants to show.

---

## 8. 2026-10-18: Build for TintKin (user said "do all"). Backups: `*.bak_step1`, `skinai/scoring.py.bak`, `app.py.bak`, `models/calibration_step3_stopgap.json`

### 8a. Data flow of the new pipeline
`app.py` (`/v1/skin-analysis` or `/analyze/*`) -> `analyze.analyze(data, real_age, with_maps)`:
1. `face.load_image` (<= 2048 px) -> `face.detect_face` -> `Face` (landmarks, bbox, segmenter mask)
2. `quality.check_quality(face)` -> issues, stats, `confidence`, `comparable` (new: `skin_smoothed`, `_skin_detail_ratio`)
3. `metrics.compute_raw(face, return_maps)`: -> raw dict (+ per-metric maps)
   - `Face.normalized(512)` -> `_prep(f)`: `_neutralize` (skin-median colour balance), `_flatten` (quadratic lighting-gradient fit, median L = 60, same factor applied to a*/b*) -> `(skin, sel, L, a, b)`
   - measures: dark_spots, blemish (median-filtered), tone_L, tone_chroma, wrinkle, wrinkle_cheek, texture + pores (native detail face `normalized(dw)`), jaw_sag, redness (internal)
4. `scoring.score_raw(raw)` -> `wrinkles, firmness, spots, radiance, overall` (+ extras `smoothness, pores, redness, clarity`), using `FOUR` weights and `models/calibration.json`
5. `age.predict_age(face.crop_face(224, 0.15, beard_free=True))` -> face age; `scoring.skin_age(face_age, scores, real_age)` -> skin age
6. `contract.to_youcam(result)` (only for `/v1/skin-analysis`): `output[]` items `wrinkle / firmness / age_spot / radiance {ui_score, raw_score, mask_png_b64}`, `all {score}`, `skin_age {score}`, plus `extras`, `confidence`, `comparable`, `pipeline_version`, `calibration_version`, `image_size`

### 8b. Files changed or added
- `skinai/metrics.py` (rewritten): illumination normalisation, new raw measures `tone_L, tone_chroma, wrinkle_cheek, jaw_sag`, `return_maps`.
- `skinai/scoring.py` (rewritten): the four TintKin metrics (wrinkles <- wrinkle; spots <- dark_spots 0.75 + blemish 0.25; radiance <- tone_L 0.5 + tone_chroma 0.5; firmness <- wrinkle_cheek 0.4 + texture 0.3 + jaw_sag 0.3), overall = mean of four, `health`, `skin_age` (shift = (calibration median health - health) x 0.12 y/pt, capped +-5 y; with real age: 0.7 real + 0.3 face age, within +-10 y of real), `calibration_version`.
- `skinai/quality.py`: `confidence` (issue penalties), `comparable` (>= 0.7), `skin_smoothed` flag (skin/eye-lip detail ratio < 0.22).
- `skinai/analyze.py`: new signature + output, `PIPELINE_VERSION = "0.2.0"`. `skinai/contract.py`: NEW. `app.py`: `/v1/skin-analysis`, `x-api-key` check (env `SKINAI_API_KEY`), `real_age` input, model warm-up.
- `scripts/calibrate.py`: new keys + `_health_center`. `models/calibration.json`: refit on the 16 `test/` photos (PROVISIONAL; Test2.jpeg is outside it). Previous: `calibration_step3_stopgap.json`, `calibration_fairface_old.json`.
- NEW `requirements.txt`, `Dockerfile`, `.dockerignore` (CPU image for Hugging Face Spaces; NOT built or run here, docker is not installed).

### 8c. Measured results (17 photos x 8 variants: brightness +-15%, warm/cool WB, JPEG q40, noise, 3% crop shift)
- Raw within-photo spread / between-photo spread (lower = steadier), old -> new: redness 0.59->0.26, dark_spots 0.35->0.15, blemish 0.38->0.33, radiance 0.25->0.09, wrinkle 0.17->0.09, pores 0.20->0.14, texture 0.13->0.09, jaw 0.31->0.22.
- Full pipeline, mean range (max-min) over the 8 variants: skin age 3.9 y -> 1.3 y (2.4 y without real age; worst photo 7.0 -> 2.0 / 5.3), clarity-type score 22.6 -> radiance 6.1, wrinkles 16.6 -> 8.1, firmness 9.0, spots 14.7 (weakest), overall 6.8.
- Responsiveness (8 photos, edits): smoothing/blur improve scores and make skin age ~3 y younger (8/8: the filter-gaming risk); +25 dark spots lower spots 8/8; +30 fine lines lower wrinkles 8/8; low-frequency colour blotches lower radiance 6/8.
- API test (`TestClient`): auth 401 without key, `/analyze/file` and `/v1/skin-analysis` 200, mask PNGs decode, non-Cloudinary URL 400, junk image 400.

### 8d. Not done / needs the user
- Shadow mode against YouCam: needs changes inside TintKin (told not to touch it).
- Real calibration on ~100 varied selfies (+ held-out set); skin-age cap/coefficient to be loosened after that.
- Mild beauty filters are not detected (only strong smoothing); firmness is a proxy; spots/blemish remain the least stable metrics.
- `.gitignore` ignores `*.onnx`/`*.pt`: pushing this repo to a Hugging Face Space needs Git LFS (or an explicit un-ignore) for `models/age.onnx`.
