# Alpha Factor Research: ML-Based Factor Evaluation Pipeline

This repository contains a modular Python pipeline to research and evaluate the predictive power of various linear and nonlinear alpha factors using free financial data. It aims to reflect realistic constraints in quant research — no premium data feeds, no real-time execution, and runs entirely locally.

---

## 📌 Project Goals

- Research modern alpha factors using free fundamental and price data
- Implement machine learning models to extract nonlinear factor structure
- Measure factor effectiveness using forward Information Coefficient (IC)
- Simulate basic long-short strategies using top-ranked alpha signals
- Modularize the codebase for easy extension with new models, features, and datasets

---

## 📂 Project Structure

```bash
.
├── data/
│   └── download.py              # Download and preprocess price & fundamental data
│
├── factors/
│   ├── linear_factors.py        # Classic alpha factors (e.g., value, quality, momentum)
│   └── nonlinear_factors.py     # ML-based features (e.g., autoencoders, nonlinear PCA)
│
├── models/
│   ├── encoder_models.py        # Autoencoder and nonlinear dimensionality models
│   └── factor_ic_eval.py        # Factor ranking and forward IC analysis
│
├── portfolio/
│   └── simulate_portfolio.py    # Construct long-short portfolios from top factors
│
├── utils/
│   └── helpers.py               # Utilities for returns, plotting, evaluation, etc.
│
├── main_pipeline.py             # End-to-end script: from data to factor IC results
└── README.md
