import requests

API_URL = "http://192.168.168.5:8000/search_pipeline_v2"  # Adjust if necessary

def test_search_pipeline(
    question: str,
    k: int = 128,
    bge_threshold: float = 0.25,
    semantic_threshold: float = 0.25
):
    payload = {
        "question": question,
        "k": k,
        "bge_threshold": bge_threshold,
        "semantic_threshold": semantic_threshold
    }
    response = requests.post(API_URL, json=payload)
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return

    results = response.json()
    if not results:
        print("No results returned.")
        return

    for idx, item in enumerate(results, start=1):
        print(f"Result #{idx}")
        print(f"  Year               : {item['year']}")
        print(f"  Document Name      : {item['doc_name']}")
        print(f"  Document Type      : {item['doc_type']}")
        print(f"  Chunk Index        : {item['chunk_index']}")
        print(f"  BGE Score          : {item['bge_score']:.4f}")
        print(f"  Semantic Score     : {item['semantic_score']:.4f}")
        print("  Facts:")
        for fact in item.get("facts", []):
            print(f"    - {fact}")
        print("  Text:")
        preview = item['text'][:200]
        print(f"    {preview}{'...' if len(item['text']) > 200 else ''}")
        # Extract entities lists
        orig_entities = item.get("metadata_original", {}).get("entities", [])
        trans_entities = item.get("metadata_translated", {}).get("entities", [])
        print("  Entities Original    :", ", ".join(orig_entities) or "None")
        print("  Entities Translated  :", ", ".join(trans_entities) or "None")
        print("-" * 80)

if __name__ == "__main__":
    test_question = "locations and descriptions of old villages"
    test_search_pipeline(
        test_question,
        k=128,
        bge_threshold=0.1,
        semantic_threshold=0.25
    )
