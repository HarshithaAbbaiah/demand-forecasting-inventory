import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [items, setItems] = useState([]);
  const [selectedItem, setSelectedItem] = useState("");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/items`)
      .then((res) => res.json())
      .then((json) => {
        setItems(json.items);
        setSelectedItem(json.items[0]);
      })
      .catch(() => setError("Could not reach the API. Is the FastAPI server running?"));
  }, []);

  useEffect(() => {
    if (!selectedItem) return;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/forecast/${selectedItem}`)
      .then((res) => {
        if (!res.ok) throw new Error("Item not found");
        return res.json();
      })
      .then((json) => {
        const chartData = json.dates.map((date, i) => ({
          date: date.slice(5),
          actual: json.actual_sales[i],
          forecast: json.forecast[i],
        }));
        setData({ ...json, chartData });
        setLoading(false);
      })
      .catch(() => {
        setError("Could not load forecast for this item.");
        setLoading(false);
      });
  }, [selectedItem]);

  return (
    <div className="dashboard">
      <header className="header">
        <h1>Demand Forecast &amp; Restock Planner</h1>
        <p className="subtitle">LightGBM-powered 28-day forecast with inventory recommendations</p>
      </header>

      <div className="controls">
        <label htmlFor="item-select">Item</label>
        <select
          id="item-select"
          value={selectedItem}
          onChange={(e) => setSelectedItem(e.target.value)}
        >
          {items.map((item) => (
            <option key={item} value={item}>{item}</option>
          ))}
        </select>
      </div>

      {error && <div className="error-banner">{error}</div>}
      {loading && <div className="loading">Loading forecast...</div>}

      {data && !loading && (
        <>
          <div className="chart-card">
            <h2>Actual vs Forecasted Sales</h2>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={data.chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
                <XAxis dataKey="date" stroke="#9aa3b2" tick={{ fontSize: 12 }} />
                <YAxis stroke="#9aa3b2" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: "#1b1f27", border: "1px solid #2a2f3a" }}
                  labelStyle={{ color: "#e8e6df" }}
                />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#9aa3b2" strokeWidth={2} dot={false} name="Actual" />
                <Line type="monotone" dataKey="forecast" stroke="#e0a458" strokeWidth={2} dot={false} name="Forecast" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="plan-grid">
            <PlanCard label="Avg Daily Demand" value={data.inventory_plan.avg_daily_demand} unit="units/day" />
            <PlanCard label="Demand Volatility" value={data.inventory_plan.demand_std} unit="std dev" />
            <PlanCard label="Safety Stock" value={data.inventory_plan.safety_stock} unit="units" highlight />
            <PlanCard label="Reorder Point" value={data.inventory_plan.reorder_point} unit="units" highlight />
          </div>
        </>
      )}
    </div>
  );
}

function PlanCard({ label, value, unit, highlight }) {
  return (
    <div className={`plan-card ${highlight ? "highlight" : ""}`}>
      <span className="plan-label">{label}</span>
      <span className="plan-value">{value}</span>
      <span className="plan-unit">{unit}</span>
    </div>
  );
}

export default App;