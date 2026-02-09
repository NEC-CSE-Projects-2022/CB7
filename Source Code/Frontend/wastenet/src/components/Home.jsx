import React from "react";

const Home = () => {
  return (
    <div className="home-container">
      {/* Inline styles scoped only to this component */}
      <style>{`
        /* ---------- Background and layout ---------- */
        .home-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 70px 20px;
          min-height: 90vh;
          background: linear-gradient(135deg, #f0f6ff, #ffffff, #e8f0ff);
          background-size: 300% 300%;
          animation: gradientShift 15s ease infinite;
          overflow-x: hidden;
        }

        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }

        /* ---------- Title Styles ---------- */
        .home-title {
          font-size: 42px;
          color: #002b5b;
          font-weight: 800;
          letter-spacing: 1.2px;
          margin-bottom: 15px;
          text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.15);
          animation: fadeInDown 1s ease-in-out;
        }

        .home-highlight {
          color: #0078d7;
          background: linear-gradient(90deg, #0078d7, #00a3cc);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          font-weight: 900;
        }

        @keyframes fadeInDown {
          from { opacity: 0; transform: translateY(-25px); }
          to { opacity: 1; transform: translateY(0); }
        }

        /* ---------- Subtitle ---------- */
        .home-subtitle {
          font-size: 22px;
          color: #004080;
          margin-bottom: 25px;
          font-weight: 600;
          letter-spacing: 0.5px;
          animation: fadeIn 1.5s ease-in;
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        /* ---------- Description Card ---------- */
        .home-description {
          max-width: 850px;
          font-size: 17px;
          line-height: 1.8;
          color: #333;
          background: rgba(255, 255, 255, 0.9);
          border-left: 6px solid #0078d7;
          border-radius: 12px;
          box-shadow: 0 8px 18px rgba(0, 0, 0, 0.1);
          padding: 15px 25px;
          text-align: justify;
          margin-top: 10px;
          transition: all 0.4s ease;
        }

        .home-description:hover {
          transform: scale(1.02);
          box-shadow: 0 12px 22px rgba(0, 0, 0, 0.15);
        }

        /* ---------- Buttons ---------- */
        .home-buttons {
          display: flex;
          gap: 20px;
          margin-top: 35px;
          animation: fadeInUp 1.2s ease;
        }

        @keyframes fadeInUp {
          from { opacity: 0; transform: translateY(30px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .home-button {
          background: linear-gradient(90deg, #0078d7, #005bb5);
          color: white;
          padding: 12px 28px;
          border-radius: 8px;
          text-decoration: none;
          font-weight: 600;
          font-size: 16px;
          letter-spacing: 0.5px;
          transition: all 0.3s ease;
          box-shadow: 0 4px 10px rgba(0, 120, 215, 0.3);
        }

        .home-button:hover {
          transform: translateY(-3px);
          background: linear-gradient(90deg, #005bb5, #004080);
          box-shadow: 0 6px 16px rgba(0, 80, 180, 0.4);
        }

        /* ---------- Footer Tagline ---------- */
        .home-footer {
          margin-top: 50px;
          color: #555;
          font-size: 16px;
          font-style: italic;
          letter-spacing: 0.3px;
          animation: fadeIn 2s ease;
        }

        /* ---------- Responsive Design ---------- */
        @media (max-width: 768px) {
          .home-title {
            font-size: 30px;
          }

          .home-subtitle {
            font-size: 18px;
          }

          .home-description {
            font-size: 15px;
            padding: 18px 20px;
          }

          .home-buttons {
            flex-direction: column;
            gap: 15px;
          }
        }
      `}</style>

      <h1 className="home-title">
        Welcome to <span className="home-highlight">SmartWasteNet</span>
      </h1>

      <h2 className="home-subtitle">
        A Deep Learning Framework for Circular Economy Transition under SDG 12
      </h2>

      <p className="home-description">
        <strong>SmartWasteNet</strong> is an intelligent AI-powered system designed to help Indian cities
        move from a <em>Take–Make–Waste</em> model toward a <em>Rethink–Redesign–Reuse</em> strategy.
        This framework integrates municipal solid waste, recycling efficiency, population density, and
        sustainability indicators using advanced deep learning techniques such as <strong>Dropout MLP</strong>
        and <strong>Autoencoders</strong>. The system identifies actionable Circular Economy (CE) strategies
        to achieve <strong>responsible consumption and production</strong> in line with
        the <strong>United Nations Sustainable Development Goal 12 (SDG 12)</strong>.
      </p>

      <div className="home-buttons">
        <a href="/about" className="home-button">About Project</a>
        <a href="/validation" className="home-button">Validate Model</a>
        <a href="/objectives" className="home-button">View Objectives</a>
      </div>

      <div className="home-footer">
        “Transforming Waste into Wisdom through Artificial Intelligence”
      </div>
      <footer className="home-footer">
        <p className="team-text">
          <strong>Developed By:</strong> <br />
          Team Members — <span className="highlight">Shaik Rasheed, V.Hemanth and K.Chandra Sekar</span>
        </p>
        <p className="guide-text">
          <strong>Under the Guidance of:</strong> <br />
          <span className="highlight"> D. Venkata Reddy</span> and <span className="highlight">Dr. S. N. Tirumala Rao</span>
        </p>
      </footer>
    </div>
  );
};

export default Home;
