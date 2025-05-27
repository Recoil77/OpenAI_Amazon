import requests
import json

API_URL = "http://192.168.168.5:8000/vector_search_v2"  # Adjust if necessary

def test_vector_search(query: str, k: int = 10):
    payload = {
        "query": query,
        "k": k
    }
    response = requests.post(API_URL, json=payload)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return

    results = response.json()
    if not results:
        print("No results returned.")
        return

    for idx, item in enumerate(results, start=1):
        print(f"Result #{idx}")
        print(f"  Document ID            : {item['document_id']}")
        print(f"  Document Name          : {item['doc_name']}")
        print(f"  Year                   : {item['year']}")
        print(f"  Doc Type               : {item['doc_type']}")
        print(f"  Chunk Index            : {item['chunk_index']}")
        print(f"  Score                  : {item['score']:.4f}")
        print(f"  Text Preview           : {item['text'][:200]}{'...' if len(item['text']) > 200 else ''}")
        print("  Metadata Original:")
        print(json.dumps(item.get('metadata_original', {}), ensure_ascii=False, indent=4))
        print("  Metadata Translated:")
        print(json.dumps(item.get('metadata_translated', {}), ensure_ascii=False, indent=4))
        print("-" * 80)

if __name__ == "__main__":
    test_query = "The Amazon River expedition hardships"
    test_vector_search(test_query, k=10)