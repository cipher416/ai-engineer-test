import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_csv(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    total = 0
    earliest = latest = None
    countries: Counter[str] = Counter()
    years: Counter[str] = Counter()
    with pd.read_csv(
        source,
        usecols=["Country", "Subscription Date"],
        chunksize=10_000,
        dtype=pd.StringDtype(storage="python"),
        keep_default_na=False,
    ) as chunks:
        for frame in chunks:
            total += len(frame)
            country = frame["Country"].str.strip()
            countries.update(country[country.ne("")].value_counts().to_dict())
            dates = pd.to_datetime(
                frame["Subscription Date"].str.strip(), format="%Y-%m-%d"
            ).dropna()
            if not dates.empty:
                first, last = dates.min().date().isoformat(), dates.max().date().isoformat()
                earliest = first if earliest is None else min(earliest, first)
                latest = last if latest is None else max(latest, last)
                years.update({str(year): count for year, count in dates.dt.year.value_counts().items()})
    return {
        "file": str(source),
        "total_rows": total,
        "subscription_dates": {
            "earliest": earliest,
            "latest": latest,
            "by_year": dict(sorted(years.items())),
        },
        "distributions": {
            "Country": {
                "distinct_values": len(countries),
                "top": [
                    {"value": value, "count": count}
                    for value, count in sorted(countries.items(), key=lambda item: (-item[1], item[0]))[:10]
                ],
            }
        },
    }


def main() -> None:
    data = Path(__file__).resolve().parents[1] / "data"
    reports = []
    for name in ("customers-source-1.csv", "customers-2000000.csv"):
        path = data / name
        try:
            reports.append(analyze_csv(path))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"CSV analysis failed for {path}: {exc}") from exc
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
