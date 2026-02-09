import React from "react";

const Procedure = () => {
  return (
    <div className="procedure-container">
      {/* Inline CSS only for this component */}
      <style>{`
        .procedure-container {
          max-width: 900px;
          margin: 40px auto;
          padding: 40px 20px;
          background: #ffffff;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          line-height: 1.7;
        }

        .procedure-title {
          text-align: center;
          color: #003366;
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 20px;
        }

        .procedure-subtitle {
          color: #003366;
          font-size: 20px;
          margin-top: 30px;
          margin-bottom: 10px;
          font-weight: 600;
        }

        .procedure-text {
          font-size: 16px;
          color: #333;
          margin-bottom: 15px;
          text-align: justify;
        }

        .procedure-list {
          padding-left: 25px;
          color: #333;
          font-size: 16px;
        }

        .procedure-list li {
          margin-bottom: 10px;
        }

        @media (max-width: 768px) {
          .procedure-container {
            margin: 20px;
            padding: 25px 15px;
          }

          .procedure-title {
            font-size: 24px;
          }

          .procedure-subtitle {
            font-size: 18px;
          }

          .procedure-text,
          .procedure-list {
            font-size: 15px;
          }
        }
      `}</style>

      <h1 className="procedure-title">Project Methodology</h1>

      <p className="procedure-text">
        The <strong>SmartWasteNet</strong> framework follows a structured and data-driven
        methodology to predict <em>Circular Economy (CE)</em> actions for urban waste
        management. The workflow integrates data collection, preprocessing, model training,
        clustering, and explainability to deliver accurate and interpretable insights.
      </p>

      <h2 className="procedure-subtitle">Step 1: Data Acquisition & Integration</h2>
      <p className="procedure-text">
        Data was collected from multiple sources, including municipal solid waste records,
        SDG 12 indicators, population and area statistics, recycling rates, and landfill
        data for <strong>59 Indian cities</strong>. These datasets were merged to form a
        comprehensive repository for circular economy analysis.
      </p>

      <h2 className="procedure-subtitle">Step 2: Data Preprocessing</h2>
      <ul className="procedure-list">
        <li>Handling missing and inconsistent values.</li>
        <li>Encoding categorical features using <strong>LabelEncoder</strong>.</li>
        <li>Feature scaling with <strong>StandardScaler</strong> to normalize input variables.</li>
        <li>Splitting the dataset into training and validation subsets.</li>
      </ul>

      <h2 className="procedure-subtitle">Step 3: Model Development</h2>
      <p className="procedure-text">
        Five deep learning models were designed and trained using <strong>TensorFlow</strong>:
      </p>
      <ul className="procedure-list">
        <li>MLP (1-layer)</li>
        <li>MLP (2-layer)</li>
        <li>Dropout MLP</li>
        <li>BatchNorm MLP</li>
        <li>WideDeep MLP</li>
      </ul>
      <p className="procedure-text">
        Each model was trained to predict the appropriate <strong>CE Action</strong> —
        Rethink, Redesign, or Reuse — based on city-level indicators. The
        <strong> Dropout MLP</strong> achieved the highest accuracy of{" "}
        <strong>97.86%</strong>.
      </p>

      <h2 className="procedure-subtitle">Step 4: Clustering with Autoencoder</h2>
      <p className="procedure-text">
        An <strong>Autoencoder</strong> network was used for dimensionality reduction and
        feature learning. The latent representations of cities were visualized in a 2D
        space, revealing natural clusters that correspond to different CE strategies.
      </p>

      <h2 className="procedure-subtitle">Step 5: Model Evaluation & Explainability</h2>
      <p className="procedure-text">
        Model performance was evaluated using accuracy, confusion matrices, and
        classification reports. <strong>SHAP (SHapley Additive exPlanations)</strong> was
        employed to interpret feature importance and understand how each input variable
        contributed to CE action prediction.
      </p>

      <h2 className="procedure-subtitle">Step 6: Action Recommendation</h2>
      <p className="procedure-text">
        Based on model predictions and clustering results, each city was assigned one of
        the CE actions — <strong>Rethink</strong>, <strong>Redesign</strong>, or
        <strong> Reuse</strong>. These recommendations help local authorities and
        policymakers adopt tailored waste management strategies aligned with
        <strong> SDG 12</strong>.
      </p>

      <p className="procedure-text">
        This stepwise pipeline ensures the <strong>SmartWasteNet</strong> framework remains
        transparent, reproducible, and adaptable for different datasets and regional contexts.
      </p>
    </div>
  );
};

export default Procedure;
