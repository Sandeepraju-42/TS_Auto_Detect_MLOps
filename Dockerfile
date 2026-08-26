# Serves app.py -- step 10 of the plan. Only what the API actually needs at
# runtime goes in here (not the whole repo): the app itself, its
# dependencies, and a local-file fallback model+schema for when the DagsHub
# registry can't be reached at startup (see app.py's load_model()).
FROM python:3.11-slim

WORKDIR /app

# libgomp1 -- LightGBM's compiled extension needs OpenMP at runtime; without
# it the import itself fails on a slim base image (nothing else here needs
# a C toolchain, so this is the only apt dependency).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY Output/tuned_lightgbm.txt Output/tuned_lightgbm.txt
COPY feature_schema_tuned.json .

EXPOSE 5000

# DagsHub auth (DAGSHUB_USER_TOKEN or equivalent) is injected as a runtime
# environment variable / secret by whatever deploys this image -- never
# baked into the image itself.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
