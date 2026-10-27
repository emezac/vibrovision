# Multi-arch image (linux/amd64 and linux/arm64 for AWS Graviton).
#   docker buildx build --platform linux/arm64 -t vibrovision:arm64 -f infra/Dockerfile .
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENCV_FOR_THREADS_NUM=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib WORK_DIR=/tmp/vibrovision
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && useradd --create-home --uid 10001 app
COPY vibrovision/ vibrovision/
COPY service/ service/
COPY bench/ bench/
COPY data/real/ data/real/
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
