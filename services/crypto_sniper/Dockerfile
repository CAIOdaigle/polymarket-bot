FROM python:3.11-slim

WORKDIR /app

# Install shared library first (layer caching)
COPY polymarket_common/ /app/polymarket_common/
RUN pip install --no-cache-dir /app/polymarket_common

# Install crypto_sniper service
COPY services/crypto_sniper/pyproject.toml /app/services/crypto_sniper/
COPY services/crypto_sniper/crypto_sniper/ /app/services/crypto_sniper/crypto_sniper/
COPY services/crypto_sniper/config/ /app/services/crypto_sniper/config/
RUN pip install --no-cache-dir /app/services/crypto_sniper

RUN useradd -m -u 1000 sniperuser && \
    mkdir -p /app/data /app/logs && \
    chown -R sniperuser:sniperuser /app/data /app/logs

USER sniperuser

VOLUME ["/app/data", "/app/logs"]

ENV SNIPER_ENV=production
ENV PYTHONPATH=/app/services/crypto_sniper

CMD ["python", "-m", "crypto_sniper.main"]
