FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/src
ENV MODEL_DIR=/app/models

RUN useradd --create-home --uid 10001 appuser

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY src/ ./src/
COPY models/ ./models/

USER appuser

# Documenter le port utilisé
EXPOSE 8000

# Vérifier la santé de l'application sans dépendre de curl
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"]

# Lancer l'API
CMD ["uvicorn", "velov.api.main:app", "--host", "0.0.0.0", "--port", "8000"]