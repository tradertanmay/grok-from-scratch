FROM docker:27-cli AS dockercli

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=8000 GROK_DB_PATH=/data/grok_bot.sqlite3 \
    GROK_PROVIDERS_FILE=/config/providers.json
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
WORKDIR /app
COPY pyproject.toml README.md ./
COPY grok_bot ./grok_bot
RUN pip install --no-cache-dir ".[mcp]" && useradd --system --uid 10001 --home /data grok \
    && mkdir -p /data /config && chown grok /data
COPY frontend ./frontend
USER grok
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health',timeout=4)"
CMD ["python", "-m", "grok_bot"]
