import requests

BASE = "http://localhost:8000"

# List of test questions for the general_knowledge endpoint
test_questions = [
    "Who founded the city of Manaus and in which year?",
    "What are the main geographical features of the Amazon Basin?",
    "What is the significance of the Portuguese fort São José da Barra do Rio Negro?",
    "we found people near the bridge of Bakska"
]

for question in test_questions:
    print(f"\n=== Question ===\n{question}\n")
    resp = requests.post(f"{BASE}/general_knowledge", json={"question": question})
    try:
        resp.raise_for_status()
        answer = resp.json().get("knowledge_answer", "")
        print("=== Answer ===")
        print(answer)
    except requests.HTTPError as e:
        print(f"Request failed: {e} - {resp.text}")
