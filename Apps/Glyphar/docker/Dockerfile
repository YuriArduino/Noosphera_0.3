FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# -------------------------------------------------
# System dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    poppler-utils \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------------------------------
# Copy dependency files first (cache optimization)
# -------------------------------------------------
COPY pyproject.toml .
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install runtime deps
RUN pip install --no-cache-dir -r requirements.txt

# Install project itself
COPY src ./src
RUN pip install --no-cache-dir -e .

# -------------------------------------------------
# Default command (override in compose)
# -------------------------------------------------

EXPOSE 8001

CMD ["uvicorn", "glyphar.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
