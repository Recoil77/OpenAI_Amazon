import requests

API_URL = "http://localhost:8000/refine_query"

# Three test cases: short, normal, long
test_cases = {
    "short": "village on the river bank",
    "normal": "What were the causes of the decline of Manaus in the 17th century?",
    "long": (
        "During the turbulent period of colonial administration and economic instability, "
        "what factors contributed to the gradual decay and abandonment of the urban settlement "
        "known as Manaus in the early seventeenth century among other Amazonian trade hubs?"
    ),
}

for name, query in test_cases.items():
    resp = requests.post(API_URL, json={"query": query})
    resp.raise_for_status()
    refined = resp.json().get("refined_query", "")
    print(f"\n=== Test: {name.upper()} ===")
    print(f"Original ({len(query.split())} words): {query}")
    print(f"Refined: {refined}")
