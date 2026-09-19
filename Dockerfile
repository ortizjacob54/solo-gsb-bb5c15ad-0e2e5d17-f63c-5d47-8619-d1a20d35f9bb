# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code and static frontend.  Keep the same directory depth as the
# repo (/app/backend/app/main.py, /app/frontend, /app/data), since main.py
# derives those paths from __file__.
COPY backend ./backend
COPY frontend ./frontend

RUN mkdir -p /app/data/images /app/data/sessions

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# --app-dir puts /app/backend on sys.path while __file__ stays /app/backend/app/main.py
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
