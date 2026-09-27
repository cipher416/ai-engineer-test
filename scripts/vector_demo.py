import json
import os
import sys
from pathlib import Path

from pydantic import TypeAdapter
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.vectors import cosine_similarity

COLLECTION = "assessment_cosine_demo"


def main():
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=10)
    try:
        if not client.collection_exists(COLLECTION):
            _ = client.create_collection(
                COLLECTION, vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT)
            )
        vectors = [[1.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        _ = client.upsert(COLLECTION, [models.PointStruct(id=i, vector=v) for i, v in enumerate(vectors, 1)])
        points = client.retrieve(COLLECTION, ids=[1, 2, 3, 4], with_vectors=True)
        vector_adapter = TypeAdapter(list[float])
        ranked = sorted(
            [(cosine_similarity([1, 0, 0], vector_adapter.validate_python(point.vector, strict=True)), point.id)
             for point in points],
            key=lambda pair: (-pair[0], pair[1]),
        )
        result = [{"id": point_id, "cosine": score} for score, point_id in ranked]
        assert [row["id"] for row in result] == [1, 2, 3, 4], result
        print(json.dumps(result, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
