import pandas as pd
import yfinance as yf
import requests
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- STEP 1: Get S&P 500 tickers ----------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(response.text)

    for table in tables:
        for col in table.columns:
            if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                tickers = table[col].astype(str).str.replace(".", "-", regex=False).tolist()
                return tickers
    raise ValueError("No Symbol/Ticker column found in Wikipedia table.")

tickers = get_sp500_tickers()
print(f"✅ Loaded {len(tickers)} S&P 500 tickers")

# ---------- STEP 2: Batch bulk downloader ----------
def compute_changes(batch):
    try:
        data = yf.download(batch, period="6mo", interval="1d", group_by="ticker", progress=False)
        results = []
        for ticker in batch:
            try:
                close = data[ticker]["Close"]
                start_price, end_price = close.iloc[0], close.iloc[-1]
                if pd.notna(start_price) and pd.notna(end_price) and start_price != 0:
                    change = (end_price - start_price) / start_price * 100
                    results.append({"Symbol": ticker, "Change_6M_%": round(float(change), 2)})
            except Exception as e:
                pass
        return results
    except Exception as e:
        print(f"⚠️ Batch failed: {batch[:5]}... | {e}")
        return []

# ---------- STEP 3: Parallel batching ----------
BATCH_SIZE = 50
MAX_THREADS = 5

batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
results = []

with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = {executor.submit(compute_changes, b): b for b in batches}
    for future in as_completed(futures):
        batch_results = future.result()
        if batch_results:
            results.extend(batch_results)

# ---------- STEP 4: Clean + sort ----------
if not results:
    raise RuntimeError("No data returned. Possibly network or Yahoo throttling issue.")

df = pd.DataFrame(results)
df = df.dropna(subset=["Change_6M_%"])
df["Change_6M_%"] = df["Change_6M_%"].astype(float)
df = df.sort_values("Change_6M_%", ascending=False)

print("\n🏆 Top 20 S&P 500 Gainers (6 Months):")
print(df.head(20).to_string(index=False))
