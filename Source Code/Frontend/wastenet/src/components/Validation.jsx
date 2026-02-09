import React, { useState } from "react";
import axios from "axios";
import "./Validation.css";

const CITY_LIST = [
  "Delhi", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur",
  "Lucknow", "Surat", "Kanpur", "Nagpur", "Patna", "Bhopal", "Indore",
  "Vadodara", "Guwahati", "Coimbatore", "Ranchi", "Amritsar", "Varanasi",
  "Ludhiana", "Agra", "Meerut", "Nashik", "Rajkot", "Madurai", "Jabalpur",
  "Allahabad"
];

export default function Validation() {
  const [selectedCity, setSelectedCity] = useState("");
  const [forecast, setForecast] = useState(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const handleForecast = async () => {
    if (!selectedCity) {
      alert("Please select a city first.");
      return;
    }

    setLoading(true);
    setStatus("Fetching forecast...");
    setForecast(null);

    try {
      const res = await axios.post("http://127.0.0.1:5000/api/forecast", {
        city: selectedCity,
      });

      if (res.data.success) {
        setForecast(res.data);
        setStatus("✔ Prediction generated successfully!");
      } else {
        setStatus(`⚠ ${res.data.message}`);
      }
    } catch (error) {
      console.error(error);
      setStatus("❌ Error connecting to backend.");
    }

    setLoading(false);
  };

  const getActionColor = (action) => {
    if (!action) return "";
    const a = action.toLowerCase();
    if (a === "rethink") return "rethink-card";
    if (a === "redesign") return "redesign-card";
    if (a === "reuse") return "reuse-card";
    return "";
  };

  return (
    <div className="validation-container">
      <h1 className="page-title">SmartWasteNet — CE Action Predictor</h1>
      <p className="subtitle">AI-powered Circular Economy Forecasting for Indian Cities</p>

      <div className="header-stats">
        <div className="accuracy-card">
          <h3>📊 Dropout MLP Accuracy</h3>
          <p className="accuracy-value">97.86%</p>
          <span className="accuracy-badge">Best Performing Model</span>
        </div>

        <div className="city-card">
          <h3>🏙 Select City for Prediction</h3>
          <select
            className="dropdown"
            value={selectedCity}
            onChange={(e) => setSelectedCity(e.target.value)}
          >
            <option value="">-- Choose City --</option>
            {CITY_LIST.map((city, idx) => (
              <option key={idx} value={city}>{city}</option>
            ))}
          </select>

          <button
            className="btn-forecast"
            onClick={handleForecast}
            disabled={!selectedCity || loading}
          >
            {loading ? "Predicting..." : "Get Forecast"}
          </button>

          {status && <p className="status-text">{status}</p>}
        </div>
      </div>

      {forecast && (
        <div className={`result-card ${getActionColor(forecast.circular_economy_action)}`}>
          <h2 className="result-city">{forecast.city}</h2>
          <p className="action-text">
            <strong>Predicted CE Action: </strong> {forecast.circular_economy_action}
          </p>

          <div className="stats-grid">
            <div><strong>Current Waste:</strong> {forecast.waste_forecast.current_waste} TPD</div>
            <div><strong>Forecasted Waste:</strong> {forecast.waste_forecast.predicted_waste} TPD</div>
            <div><strong>Growth Rate:</strong> {forecast.waste_forecast.growth_rate}%</div>
            <div><strong>Recycling Rate:</strong> {forecast.recycling_rate}%</div>
            <div><strong>Municipal Efficiency:</strong> {forecast.municipal_efficiency}/10</div>
            <div><strong>Population Density:</strong> {forecast.population_density} people/km²</div>
            <div><strong>SDG Score:</strong> {forecast.sdg_score}</div>
          </div>

          <h3 className="recommend-title">Recommendations</h3>
          <ul className="recommend-list">
            {forecast.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
