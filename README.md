---
title: Tintkin Skin Analysis
emoji: 🧴
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

CPU skin-analysis API for TintKin. `POST /v1/skin-analysis` with `{"image_url": "<Cloudinary URL>", "real_age": 24, "masks": true}`.

Space secrets: `SKINAI_API_KEY` (required header `x-api-key`), `CLOUDINARY_CLOUD_NAME` (restricts image URLs to that cloud).
