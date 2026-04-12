FROM python:3.11-slim

WORKDIR /app

# Install shared library first (layer caching)
COPY polymarket_common/ /app/polymarket_common/
RUN pip install --no-cache-dir /app/polymarket_common

# Install the main bot
COPY pyproject.toml .
COPY config/ config/
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

RUN useradd -m -u 1000 botuser && \
    mkdir -p /app/data /app/logs && \
    chown -R botuser:botuser /app/data /app/logs

USER botuser

VOLUME ["/app/data", "/app/logs"]

EXPOSE 5050

COPY start.sh .

ENTRYPOINT ["sh", "start.sh"]
