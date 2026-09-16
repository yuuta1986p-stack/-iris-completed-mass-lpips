FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Warm the AlexNet/LPIPS weights into the image at build time.
RUN python -c "import lpips; lpips.LPIPS(net='alex').eval(); print('LPIPS AlexNet ready')"
COPY app ./app
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]
