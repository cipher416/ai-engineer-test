import os
from datetime import date
from pathlib import Path

from pydantic import TypeAdapter

from app.receipts import ReceiptCreate
from app.store import Store


def main() -> None:
    source = Path(__file__).resolve().parents[1] / "data/examples/receipts.json"
    examples = TypeAdapter(list[ReceiptCreate]).validate_json(source.read_text())
    store = Store(os.getenv("DB_PATH", "data/receipts.sqlite3"))
    existing = {receipt["raw_text"] for receipt in store.query(date.min, date.max)["receipts"]}
    added = 0
    for receipt in examples:
        if receipt.raw_text not in existing:
            store.save(receipt)
            existing.add(receipt.raw_text)
            added += 1
    print(f"Loaded {added} synthetic example receipts; skipped {len(examples) - added} already present.")


if __name__ == "__main__":
    main()
