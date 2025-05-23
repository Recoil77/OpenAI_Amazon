import requests

# Base URL for the web_search endpoint
BASE_URL = "http://192.168.168.5:8100"

# 1) Define test queries
test_queries = [
    "Какая сейчас погода в Белграде?",
    "Последние новости о исследовании Амазонки",
    "Исторические события, связанные с рекой Погубу"
]

# 2) Call the web_search endpoint for each query
for query in test_queries:
    payload = {"query": query, "recency_days": 7}
    try:
        resp = requests.post(f"{BASE_URL}/web_search", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        print(f"\n=== Query: {query} ===")
        print(f"Answer:\n{data['answer']}\n")
        if data.get('sources'):
            print("Sources:")
            for src in data['sources']:
                print(f"  - {src}")
        else:
            print("Sources: (none returned)")
    except requests.RequestException as e:
        print(f"Error for query '{query}': {e}")
