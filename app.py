import os
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ✅ Enable CORS for your React frontend (adjust origin if needed)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# --- Chart logic ---
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


# --- API Routes ---
@app.route("/api/breakouts")
def get_breakouts():
    symbols_file = "datasets/symbols.csv"
    daily_folder = "datasets/daily"

    if not os.path.exists(symbols_file):
        return jsonify({"error": "symbols.csv not found"}), 404

    symbols_df = pd.read_csv(symbols_file, sep="\t", header=None, names=["Symbol", "Company"])
    breakouts = []

    for _, row in symbols_df.iterrows():
        symbol = row["Symbol"]
        company = row["Company"]
        csv_path = os.path.join(daily_folder, f"{symbol}.csv")

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if df.empty or "Close" not in df.columns:
            continue

        if is_breaking_out(df):
            breakouts.append({
                "symbol": symbol,
                "company": company,
                "chart": f"https://finviz.com/chart.ashx?t={symbol}&ty=c&ta=1&p=d&s=l"
            })

    return jsonify({"count": len(breakouts), "breakouts": breakouts})


@app.route("/api/snapshot")
def snapshot():
    # placeholder for future functionality if needed
    return jsonify({"code": "success"})


# --- Main entry point for local testing ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


