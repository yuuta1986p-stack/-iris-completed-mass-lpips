FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# AlexNet / LPIPS weights をビルド時に取得
RUN python -c "import lpips; lpips.LPIPS(net='alex')"

COPY app ./app

EXPOSE 8000

CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
