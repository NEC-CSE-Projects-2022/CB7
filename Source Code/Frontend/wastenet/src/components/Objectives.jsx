

import React from "react";

const Objectives = () => {
  return (
    <div className="objectives-container">
      {/* Inline CSS styles only for this component */}
      <style>{`
        .objectives-container {
          max-width: 900px;
          margin: 40px auto;
          padding: 40px 20px;
          background: #ffffff;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          line-height: 1.7;
        }

        .objectives-title {
          text-align: center;
          color: #003366;
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 20px;
        }

        .objectives-text {
          font-size: 16px;
          color: #333;
          margin-bottom: 20px;
          text-align: justify;
        }

        .objectives-list {
          padding-left: 25px;
          color: #333;
          font-size: 16px;
        }

        .objectives-list li {
          margin-bottom: 10px;
        }

        @media (max-width: 768px) {
          .objectives-container {
            margin: 20px;
            padding: 25px 15px;
          }

          .objectives-title {
            font-size: 24px;
          }

          .objectives-text,
          .objectives-list {
            font-size: 15px;
          }
        }
      `}</style>

      <h1 className="objectives-title">Project Objectives</h1>

      <p className="objectives-text">
        The primary aim of <strong>SmartWasteNet</strong> is to develop an intelligent
        deep learning framework that assists Indian cities in achieving a
        <strong> Circular Economy (CE)</strong> transition in line with
        <strong> SDG 12 – Responsible Consumption and Production</strong>.
        The framework leverages data-driven insights to recommend
        appropriate CE actions — <em>Rethink</em>, <em>Redesign</em>, and <em>Reuse</em> —
        for sustainable solid waste management.
      </p>

      <h2 className="objectives-title" style={{ fontSize: "22px", marginTop: "30px" }}>
        Specific Objectives
      </h2>

      <ul className="objectives-list">
        <li>
          📊 <strong>Data Integration:</strong> Combine multi-source datasets including
          municipal solid waste, recycling efficiency, SDG indicators, and population
          characteristics for 59 Indian cities.
        </li>
        <li>
          🧠 <strong>Model Development:</strong> Design and train multiple deep learning
          architectures (MLP, Dropout MLP, BatchNorm MLP, and WideDeep MLP) to predict
          optimal CE actions based on city-level indicators.
        </li>
        <li>
          🔍 <strong>Feature Learning:</strong> Employ an autoencoder-based approach to
          extract latent features and cluster cities with similar CE behavior patterns.
        </li>
        <li>
          🎯 <strong>Performance Evaluation:</strong> Evaluate models using accuracy and
          classification metrics to select the best-performing architecture (Dropout MLP
          with <strong>97.86% accuracy</strong>).
        </li>
        <li>
          🌱 <strong>Decision Support:</strong> Provide interpretable and actionable
          insights to policymakers and urban planners for implementing sustainable CE
          strategies tailored to each city cluster.
        </li>
        <li>
          ♻️ <strong>Sustainability Impact:</strong> Promote the adoption of AI-based
          frameworks in solid waste management to minimize landfill dependency and
          maximize recycling and reuse initiatives.
        </li>
      </ul>
    </div>
  );
};

export default Objectives;
