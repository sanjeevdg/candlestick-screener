import { createChart } from 'lightweight-charts';
import { useEffect, useRef, useState } from 'react';

const indicators = ['RSI', 'SMA', 'EMA', 'MACD'];

export default function StockChart({ symbol }) {
  const chartRef = useRef(null);
  const [indicator, setIndicator] = useState('RSI');

  useEffect(() => {
    const chart = createChart(chartRef.current, {
      height: 600,
      layout: { background: { color: '#0b0e11' }, textColor: '#DDD' },
      grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } }
    });

    const candleSeries = chart.addSeries('Candlestick');
    const volumeSeries = chart.addSeries('Histogram', {
      priceFormat: { type: 'volume' },
      priceScaleId: ''
    });

    const indicatorSeries = chart.addSeries('Line', {
      priceScaleId: 'right'
    });

    fetch(`/api/chart?symbol=${symbol}&indicator=${indicator}`)
      .then(res => res.json())
      .then(data => {
        candleSeries.setData(data.candles);
        volumeSeries.setData(data.volume);
        indicatorSeries.setData(data.indicatorSeries);
      });

    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [symbol, indicator]);

  return (
    <>
      <select
        value={indicator}
        onChange={(e) => setIndicator(e.target.value)}
        style={{ marginBottom: 10 }}
      >
        {indicators.map(i => (
          <option key={i} value={i}>{i}</option>
        ))}
      </select>

      <div ref={chartRef} />
    </>
  );
}
