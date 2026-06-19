import requests
import pandas as pd

url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}

response = requests.get(url, headers=headers, timeout=10)

if response.status_code != 200:
    print(f"❌ HTTP error {response.status_code}")
    print(response.text[:500])
    exit()

try:
    data = response.json()
except Exception:
    print("❌ Failed to parse JSON. Returned text snippet:")
    print(response.text[:500])
    exit()

quotes = data["finance"]["result"][0]["quotes"]

def safe_value(v):
    """Return raw number if dict, otherwise v itself"""
    if isinstance(v, dict):
        return v.get("raw", v.get("fmt", None))
    return v

rows = []
for q in quotes:
    rows.append({
        "symbol": q.get("symbol"),
        "name": q.get("shortName"),
        "price": safe_value(q.get("regularMarketPrice")),
        "change": safe_value(q.get("regularMarketChange")),
        "percent_change": safe_value(q.get("regularMarketChangePercent")),
        "volume": safe_value(q.get("regularMarketVolume")),
    })

df = pd.DataFrame(rows)
print(df.head())
