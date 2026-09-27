# AI Solution Engineer assessment

Customer CSV analysis, handwritten cosine similarity with Qdrant, and a Streamlit receipt assistant using LangChain and SQLite. [Engineering answers](docs/engineering-answers.md).

## Demo Video

https://github.com/user-attachments/assets/508ef5a8-f328-4a3c-bcd5-522a4d6367d2

## Run

Requires Docker Compose and an OpenRouter API key. Choose a model that supports images, structured output and tool calls.

```sh
cp .env.example .env
# Set OPENROUTER_API_KEY and LLM_MODEL in .env.
docker compose up --build
```

Open [localhost:8501](http://localhost:8501). Upload the [synthetic receipt](data/examples/corner-kitchen.png), extract its fields, review and save, then ask questions in **Ask**. Images and chat requests go through the configured provider.

To load the five [fictional example records](data/examples/receipts.json) directly:

```sh
docker compose exec -T app python -m scripts.load_examples
```

Existing records are preserved and repeated imports are skipped. Dates are fixed: try “What did I buy on September 26, 2026?” or “How much did I spend on June 20, 2026?” Uploading and saving the same receipt after importing it creates a duplicate purchase.

Data persists in Docker volumes. The app is intended for local use and has no authentication.

## CSV analysis

Download the assessment datasets to:

- [100,000-row CSV](https://drive.google.com/uc?id=1N1xoxgcw2K3d-49tlchXAWw4wuxLj7EV&export=download) → `data/customers-source-1.csv`
- [2,000,000-row ZIP](https://drive.google.com/uc?id=1IXQDp8Um3d-o7ysZLxkDyuvFj9gtlxqz&export=download) → extract as `data/customers-2000000.csv`

With Python 3.12+ and uv:

```sh
uv sync --frozen
uv run python -m app.csv_analysis
```

Prints row counts, country distributions and subscription-date summaries for both files. Both cover 2020 through May 2022, so the lower 2022 count represents a partial year.

Small files can fit in one DataFrame; this implementation reads 10,000-row chunks and retains only country/year counts and date bounds. Memory depends on chunk size and distinct aggregate keys. Physical file splitting is unnecessary here; if splitting for parallel work, preserve CSV record boundaries and headers, then merge counts and date bounds.

## Vector demo

With Compose running:

```sh
docker compose exec -T app python scripts/vector_demo.py
```

Stores four vectors in Qdrant and ranks them using the [handwritten cosine implementation](app/vectors.py). Qdrant handles storage; similarity is calculated in Python.

## Checks

```sh
uv run ruff check .
```
