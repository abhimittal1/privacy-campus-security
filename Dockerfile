# ── Stage 1 – runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Abhishek Mittal & Smriti Aggarwal"
LABEL description="Privacy-Preserving Campus Security – Flask/SocketIO dashboard"

WORKDIR /app

# ── System deps ─────────────────────────────────────────────────────────────────
# slim base has no C libs; Pillow and cryptography wheels are self-contained,
# but we still need ca-certificates for HTTPS model weight downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (cached unless requirements.txt changes) ─────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Install the package itself (registers the `pcs` entry-point) ─────────────────
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e . --no-deps

# ── Copy remaining project files ─────────────────────────────────────────────────
COPY . .

# ── Runtime data directory (bind-mounted from the host in production) ────────────
RUN mkdir -p /app/data

# ── Default environment variables ────────────────────────────────────────────────
ENV PCS_STORE=/app/data/embeddings.json \
    PCS_ENCRYPT=0 \
    PCS_DEVICE=cpu \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=0

EXPOSE 5000

# ── Entry point ──────────────────────────────────────────────────────────────────
CMD ["python", "src/web/flask_app.py"]
