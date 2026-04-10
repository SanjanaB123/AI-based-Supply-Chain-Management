FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY routes_inventory.py .
COPY routes_chat.py .
COPY models.py .
COPY database.py .
COPY clerk_auth.py .
COPY training/ ./training/
COPY config/ ./config/

# Train SetFit classifier at build time — model saved to config/inventory_query_classifier
RUN python training/train_router.py

# Run as non-root user
RUN adduser --disabled-password --no-create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
