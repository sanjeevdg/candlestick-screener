# scanner/momentum.py
print("STARTING momentum.py")
from alpaca.data.requests import StockSnapshotRequest
from alpaca.data.requests import StockSnapshotRequest


print("IMPORT SUCCESS")


def clean_symbols(symbols):
    cleaned = []
    for s in symbols:
        s = s.replace(".", "-")

        # Skip known problematic patterns
        if "-" in s and not s.endswith("-USD"):
            continue

        cleaned.append(s)

    return cleaned





def fetch_snapshots_safe(data_client, symbols, chunk_size=50):
    all_snaps = {}

    symbols = clean_symbols(symbols)

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]

        try:
            request = StockSnapshotRequest(symbol_or_symbols=chunk)
            snaps = data_client.get_stock_snapshot(request)

            if snaps:
                all_snaps.update(snaps)

        except Exception as e:
            print(f"Skipping chunk due to error: {e}")
            continue

    return all_snaps



def get_top_momentum(data_client, symbols, limit=15):
	
    symbols = [s.replace(".", "-") for s in symbols]
    
    request = StockSnapshotRequest(symbol_or_symbols=symbols)
    snapshots = fetch_snapshots_safe(data_client, symbols)
    #print("SNAPSHOTS TYPE:", type(snapshots))
    print("SNAPSHOTS:", snapshots)
    print("LEN SNAPSHOTS:", len(snapshots))
    candidates = []
    fail_reasons = {
        "no_data": 0,
        "price": 0,
        "volume": 0,
        "range": 0,
        "valid": 0
    }
    for symbol in snapshots:
        snap = snapshots[symbol]
        
        #print("SNAP:", snap)
        try:
            if not snap or not snap.latest_trade:
                fail_reasons["no_data"] += 1
                continue
            print("Processing:", symbol)    
            latest_trade = getattr(snap, "latest_trade", None)
            daily_bar = getattr(snap, "daily_bar", None)
            prev_bar = getattr(snap, "previous_daily_bar", None)

            if not daily_bar or not prev_bar:
            	fail_reasons["no_data"] += 1
    	        continue

            price = getattr(latest_trade, "price", None)
            if price is None:
    	        fail_reasons["no_data"] += 1
    	        continue
            prev_close = getattr(prev_bar, "close", None) if prev_bar else None
            prev_volume = getattr(prev_bar, "volume", None) if prev_bar else None    	        
            #prev_close = prev_bar.close

            volume = daily_bar.volume
            #prev_volume = prev_bar.volume
            if prev_close is None:
                prev_close = price   # avoids crash, neutral change

            if prev_volume is None:
                prev_volume = volume  # avoids division issues

            if not price or not prev_close:
                fail_reasons["no_data"] += 1
                continue

            # --- BASIC FILTERS ---
            if price < 2:
                fail_reasons["price"] += 1
                continue

            if volume < 200_000:
                fail_reasons["volume"] += 1
                continue

            # --- METRICS ---
            change_pct = (price - prev_close) / prev_close * 100

            relative_volume = volume / max(prev_volume, 1)

            vwap = getattr(daily_bar, "vwap", None) or price

            range_pct = (daily_bar.high - daily_bar.low) / price * 100

            # Avoid dead stocks
            if range_pct < 0.5:
                fail_reasons["range"] += 1
                continue
            fail_reasons["valid"] += 1    
            # --- BREAKOUT LOGIC ---
            breakout_bonus = 0
            if price > daily_bar.high * 0.98:
                breakout_bonus = 3

            print(symbol, price, volume) 
            # --- SCORE ---
            score = (
                change_pct * 0.5 +
                min(relative_volume, 5) * 0.3 +
                (1 if price > vwap else 0) * 5 +
                breakout_bonus
            )

            candidates.append({
                "symbol": symbol,
                "price": round(price, 2),
                "change_pct": round(change_pct, 2),
                "volume": volume,
                "rel_volume": round(relative_volume, 2),
                "score": round(score, 2)
            })

        except Exception as e:
            print(f"ERROR for {symbol}: {e}")
            continue

    # Sort descending
    top = sorted(candidates, key=lambda x: x["score"], reverse=True)
    print(f"Scanned: {len(symbols)} | Candidates: {len(candidates)}")
    print("FAIL STATS:", fail_reasons)

    return top[:limit]
