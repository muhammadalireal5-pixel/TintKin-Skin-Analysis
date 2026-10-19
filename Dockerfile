# Hugging Face Spaces (Docker, CPU). Not built on the dev machine (no Docker); app itself is tested with uvicorn.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY skinai ./skinai
COPY app.py .
COPY models/age.onnx models/calibration.json models/face_landmarker.task models/selfie_multiclass_256x256.tflite ./models/
ENV PORT=7860
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
