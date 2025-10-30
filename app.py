from flask import Flask, jsonify
from flask_cors import CORS
import os
import pandas as pd
import csv

app = Flask(__name__)
CORS(app)
# -------------------------------
# Core logic
# -------------------------------

def is_consolidating(df, percentage=2):
    if len(df) < 15:
        return False

    recent_candlesticks = df.iloc[-15:]
    max_close = recent_candlesticks['Close'].max()
    min_close = recent_candlesticks['Close'].min()
    threshold = 1 - (percentage / 100)
    return min_close > (max_close * threshold)


def is_breaking_out(df, percentage=2.5):
    if len(df) < 16:
        return False

    last_close = df.iloc[-1]['Close']

    if is_consolidating(df.iloc[:-1], percentage=percentage):
        recent_closes = df.iloc[-16:-1]
        if last_close > recent_closes['Close'].max():
            return True

    return False


# -------------------------------
# Helper: Load symbols & names
# -------------------------------
def load_symbol_names():
    symbols = {}
    symbols_path = "datasets/symbols.csv"
    if os.path.exists(symbols_path):
        with open(symbols_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    symbols[row[0].strip()] = row[1].strip()
    return symbols


# -------------------------------
# API Route
# -------------------------------
@app.route("/api/breakouts", methods=["GET"])
def get_breakouts():
    symbols_map = load_symbol_names()
    results = []

    daily_path = "datasets/daily"
    if not os.path.exists(daily_path):
        return jsonify({"error": "datasets/daily folder not found"}), 404

    for filename in os.listdir(daily_path):
        if not filename.endswith(".csv"):
            continue

        path = os.path.join(daily_path, filename)
        try:
            df = pd.read_csv(path)
            if df.empty or "Close" not in df.columns:
                continue

            symbol = filename.replace(".csv", "")
            company = symbols_map.get(symbol, "Unknown Company")

            if is_consolidating(df):
                results.append({
                    "symbol": symbol,
                    "company": company,
                    "status": "consolidating",
                    "chart": f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
                })
            elif is_breaking_out(df):
                results.append({
                    "symbol": symbol,
                    "company": company,
                    "status": "breaking_out",
                    "chart": f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
                })

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    return jsonify(results)


# -------------------------------
# Main entrypoint
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

