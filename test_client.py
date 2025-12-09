# test_client.py
import requests

API_URL = "http://127.0.0.1:8000/ask"

def main():
    print("Type a message for the assistant. Ctrl+C to exit.\n")
    while True:
        msg = input("You: ")
        resp = requests.post(API_URL, json={"message": msg})
        if not resp.ok:
            print("Request failed with status:", resp.status_code)
            print("Body:", resp.text)
            print("-" * 40)
            continue

        data = resp.json()
        print("Assistant:", data.get("answer"))
        print("-" * 40)

if __name__ == "__main__":
    main()
