FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /usr/local/bin/uv
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY scripts ./scripts
COPY data/examples ./data/examples
RUN mkdir /data && chown app:app /data
ENV PATH="/app/.venv/bin:$PATH" DB_PATH=/data/receipts.sqlite3 PYTHONUNBUFFERED=1
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"
CMD ["python", "-m", "streamlit", "run", "app/ui.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.maxUploadSize=8", "--browser.gatherUsageStats=false"]
