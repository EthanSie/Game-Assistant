import requests

API_URL = "http://127.0.0.1:8000/crawl_site"

def main():
    payload = {
        "root_url": "https://minecraft.wiki/",
        "max_pages": 30
    }

    resp = requests.post(API_URL, json=payload)
    print("Status:", resp.status_code)
    print("Response text:", resp.text)

if __name__ == "__main__":
    main()
