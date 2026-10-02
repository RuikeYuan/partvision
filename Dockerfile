FROM python:3.12-slim

WORKDIR /app
# CPU-only torch keeps the image ~1 GB instead of ~5 GB with CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY partvision ./partvision
COPY scripts ./scripts

# models/ and data/ are mounted as volumes so model versions and feedback survive redeploys
ENV PARTVISION_HOME=/app
VOLUME ["/app/models", "/app/data"]
EXPOSE 8000
CMD ["uvicorn", "partvision.api:app", "--host", "0.0.0.0", "--port", "8000"]
