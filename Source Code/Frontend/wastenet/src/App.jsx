import React from "react";
import "./App.css";
import { Routes, Route, Link } from "react-router-dom";
import Home from "./components/Home";
import About from "./components/About";
import Objectives from "./components/Objectives";
import Procedure from "./components/Procedure";
import Validation from "./components/Validation";

function App() {
  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-title">SmartWasteNet: A Deep Learning Framework to Transition from Take-Make-Waste to Rethink-Redesign-Reuse for Circular Economy under SDG 12</div>
      </header>

       {/* Navbar */}
        <nav className="navbar">
          <Link to="/">Home</Link>
          <Link to="/about">About Project</Link>
          <Link to="/objectives">Objectives</Link>
          <Link to="/procedure">Procedure</Link>
          <Link to="/validation">Validation</Link>
        </nav>

        {/* Routes */}
        <main className="content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/objectives" element={<Objectives />} />
            <Route path="/procedure" element={<Procedure />} />
            <Route path="/validation" element={<Validation />} />
          </Routes>
        </main>

      

      {/* Footer */}
      <footer className="footer">© Dept. of CSE, Narasaraopeta Engineering College,NRT,Palnadu,AP, India</footer>
    </div>
  );
}

export default App;
