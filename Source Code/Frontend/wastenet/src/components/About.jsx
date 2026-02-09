import React from "react";

const About = () => {
  return (
    <div className="about-container">
      {/* Inline CSS injected only for this component */}
      <style>{`
        .about-container {
          max-width: 900px;
          margin: 40px auto;
          padding: 40px 20px;
          background: #ffffff;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          line-height: 1.7;
        }

        .about-title {
          text-align: center;
          color: #003366;
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 20px;
        }

        .about-subtitle {
          color: #003366;
          font-size: 20px;
          margin-top: 30px;
          margin-bottom: 10px;
          font-weight: 600;
        }

        .about-text {
          font-size: 16px;
          color: #333;
          margin-bottom: 15px;
          text-align: justify;
        }

        .about-list {
          padding-left: 20px;
          margin-bottom: 15px;
          font-size: 16px;
          color: #333;
        }

        .about-list li {
          margin-bottom: 8px;
        }

        @media (max-width: 768px) {
          .about-container {
            margin: 20px;
            padding: 25px 15px;
          }

          .about-title {
            font-size: 24px;
          }

          .about-subtitle {
            font-size: 18px;
          }

          .about-text,
          .about-list {
            font-size: 15px;
          }
        }
      `}</style>

      <h1 className="about-title">About SmartWasteNet</h1>

      <p className="about-text">
        <strong>SmartWasteNet</strong> is an intelligent deep learning framework designed to
        accelerate India’s transition from the traditional{" "}
        <em>Take–Make–Waste</em> linear model to a{" "}
        <em>Rethink–Redesign–Reuse</em> circular economy, aligned with{" "}
        <strong>United Nations Sustainable Development Goal 12</strong> — 
        <em> Responsible Consumption and Production</em>.
      </p>

      <p className="about-text">
        The system integrates multi-dimensional datasets including municipal solid waste
        generation, recycling efficiency, population density, SDG 12 indicators, and landfill
        characteristics from <strong>59 Indian cities</strong> to identify and recommend
        Circular Economy (CE) actions for sustainable waste management.
      </p>

      <p className="about-text">
        SmartWasteNet employs five deep learning architectures —
        <strong> MLP (1-layer)</strong>, <strong>MLP (2-layer)</strong>,
        <strong> Dropout MLP</strong>, <strong>BatchNorm MLP</strong>, and
        <strong> WideDeep MLP</strong> — to predict CE actions. Among them, the
        <strong> Dropout MLP</strong> model achieved the highest accuracy of{" "}
        <strong>97.86%</strong>, demonstrating excellent generalization and reliability.
      </p>

      <p className="about-text">
        Using an <strong>autoencoder-based clustering mechanism</strong>, SmartWasteNet
        maps cities into a latent 2D feature space and classifies them into three key CE actions:
      </p>

      <ul className="about-list">
        <li>♻️ <strong>Rethink</strong> — optimize existing waste management strategies.</li>
        <li>🔄 <strong>Redesign</strong> — re-engineer product and material flows.</li>
        <li>🔁 <strong>Reuse</strong> — encourage recovery and reuse of resources.</li>
      </ul>

      <p className="about-text">
        This project establishes an <strong>AI-driven foundation</strong> for municipal
        authorities and policymakers to make data-supported, sustainable waste management
        decisions and promote circularity in urban systems.
      </p>

      <h2 className="about-subtitle">Key Highlights</h2>
      <ul className="about-list">
        <li>Dataset: Integrated municipal solid waste data from 59 Indian cities.</li>
        <li>Techniques: Autoencoder, MLPs, Dropout & Batch Normalization.</li>
        <li>Tools: TensorFlow 2.x, Scikit-learn, and Python 3.</li>
        <li>Performance: Dropout MLP achieved <strong>97.86% accuracy</strong>.</li>
        <li>Outcome: CE Action recommendations — <em>Rethink, Redesign, Reuse</em>.</li>
      </ul>

      <h2 className="about-subtitle">Impact</h2>
      <p className="about-text">
        SmartWasteNet demonstrates how <strong>AI and data-driven insights</strong> can
        revolutionize municipal waste management systems. It enables cities to transition
        toward a <strong>Circular Economy</strong>, contributing directly to{" "}
        <strong>SDG 12 — Responsible Consumption and Production</strong> and promoting
        sustainable urban development across India.
      </p>
    </div>
  );
};

export default About;
