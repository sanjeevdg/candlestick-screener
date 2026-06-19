import os, csv, time
import talib
import yfinance as yf
import pandas as pd
from flask import Flask, request, render_template, render_template_string
from markupsafe import escape
from chartlib import is_consolidating, is_breaking_out

from patterns import candlestick_patterns



app = Flask(__name__)




def safe_download(symbol, start="2025-01-01", end="2025-10-01", retries=3, delay=2):
    """Download symbol data with retry logic and basic error handling."""
    for attempt in range(retries):
        try:
            print(f"📈 Downloading {symbol} (attempt {attempt + 1})...")
            data = yf.download(symbol, start=start, end=end, progress=False)
            if is_consolidating(data):
                print(f"{symbol}  is consolidating!")
            else:
                print(f"{symbol} is not consolidating.")
            if not data.empty:
                print(f"✅ Got {len(data)} rows for {symbol}")
                return data
            else:
                print(f"⚠️ Empty data for {symbol}, retrying...")
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
        time.sleep(delay)
    print(f"❌ Giving up on {symbol}")
    return pandas.DataFrame()


@app.route('/breakouts')
def breakouts():
    results = []

    # ✅ Read symbols and company names
    with open('datasets/symbols.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        symbols = [(row[0].strip(), row[1].strip()) for row in reader if len(row) >= 2]

    print(f"📘 Scanning {len(symbols)} symbols...")

    for symbol, company in symbols:
        csv_path = f"datasets/daily/{symbol}.csv"
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'Close' not in df.columns:
                continue

            # Check both patterns
            if is_consolidating(df, percentage=2.5):
                results.append({"symbol": symbol, "company": company, "status": "Consolidating"})

            elif is_breaking_out(df):
                results.append({"symbol": symbol, "company": company, "status": "Breaking Out"})

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    print(f"✅ Done scanning. Found {len(results)} results.")

    # ✅ Render as HTML using Finviz charts
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Breakout & Consolidation Scanner</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f7f9fb; margin: 20px; }
            h1 { text-align: center; }
            table { width: 90%; margin: 0 auto; border-collapse: collapse; background: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            th, td { padding: 10px; border-bottom: 1px solid #ddd; text-align: center; }
            th { background: #007bff; color: white; text-transform: uppercase; letter-spacing: 1px; }
            tr:hover { background-color: #f1f1f1; }
            img { width: 280px; height: 160px; border-radius: 4px; }
            .status-Consolidating { color: #ff9800; font-weight: bold; }
            .status-Breaking { color: #4caf50; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>📊 Breakout & Consolidation Scanner</h1>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Company</th>
                    <th>Status</th>
                    <th>Chart</th>
                </tr>
            </thead>
            <tbody>
                {% for s in results %}
                <tr>
                    <td>{{ s.symbol }}</td>
                    <td>{{ s.company }}</td>
                    <td class="status-{{ s.status.split()[0] }}">{{ s.status }}</td>
                    <td>
                        <img src="https://finviz.com/chart.ashx?t={{ s.symbol }}&ty=c&ta=1&p=d&s=l" alt="{{ s.symbol }} chart" />
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """

    return render_template_string(html_template, results=results)


@app.route('/snapshot')
def snapshot():
    print("🚀 SNAPSHOT STARTED")

    os.makedirs('datasets/daily', exist_ok=True)

    symbols_path = 'datasets/symbols.csv'
    if not os.path.exists(symbols_path):
        print(f"❌ ERROR: {symbols_path} not found!")
        return {"error": "symbols.csv not found"}, 500

    # ✅ Read and clean symbols
    with open(symbols_path) as f:
        lines = [line.strip() for line in f if "," in line]

    if not lines:
        print("❌ No valid lines found in symbols.csv")
        return {"error": "symbols.csv is empty or invalid"}, 500

    # 🔹 Restrict to first 3 symbols for testing
    lines = lines[:5]

    print(f"📘 Found {len(lines)} symbols to process:")
    for i, l in enumerate(lines, 1):
        print(f"   {i}. {repr(l)}")

    for line in lines:
        symbol = line.split(",")[0].strip().replace("\ufeff", "")  # strip hidden BOMs etc.
        csv_path = f'datasets/daily/{symbol}.csv'

        if not symbol:
            print("⚠️ Skipping empty symbol line")
            continue

        # 🔸 Skip existing files
        if os.path.exists(csv_path):
            print(f"⏭️ Skipping {symbol} (already downloaded)")
            continue

        print(f"📈 Downloading {symbol}...")

        try:
            data = yf.download(symbol, start="2025-01-01", end="2025-10-01", progress=False)
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
            continue

        # 🧩 Debug print of data status
        if data is None or data.empty:
            print(f"⚠️ No data returned for {symbol}")
            continue

        print(f"✅ {symbol}: {len(data)} rows | Columns: {list(data.columns)}")

        # Save CSV
        try:
            data.to_csv(csv_path)
            print(f"💾 Saved → {csv_path}")
        except Exception as e:
            print(f"❌ Error saving {symbol}: {e}")

        time.sleep(1.5)  # polite delay to avoid rate limiting

    print("🎯 SNAPSHOT COMPLETE")
    return {"code": "success"}








@app.route('/')
def index():
    pattern = request.args.get('pattern', False)
    stocks = {}

    with open('datasets/symbols.csv') as f:
        for row in csv.reader(f):
            stocks[row[0]] = {'company': row[1]}

    if pattern:
        for filename in os.listdir('datasets/daily'):
            if not filename.endswith('.csv'):
                continue

            filepath = f'datasets/daily/{filename}'
            df = pandas.read_csv(filepath)
            symbol = filename.split('.')[0]

            try:
                pattern_function = getattr(talib, pattern)
                results = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
                last = results.tail(1).values[0]

                if last > 0:
                    stocks[symbol][pattern] = 'bullish'
                elif last < 0:
                    stocks[symbol][pattern] = 'bearish'
                else:
                    stocks[symbol][pattern] = None
            except Exception as e:
                print(f'⚠️ Failed on {filename}: {e}')

    return render_template('index.html', candlestick_patterns=candlestick_patterns, stocks=stocks, pattern=pattern)

if __name__ == "__main__":
    app.run(debug=True)

