# GDP Forecasting with Graph Neural Networks

A graph neural network with Pearson-correlation-based connectivity (**GNN–Pearson**) for
short-term (≈45-day-ahead) U.S. GDP forecasting, with baseline models for comparison
(ARIMAX, XGBoost, LSTM, Graph Transformer, and a survey-based nowcast) evaluated over an
80-quarter walk-forward window (Q1 2006 – Q4 2025).

## Repository layout

| Path | Contents |
|------|----------|
| `src/gnn_gdp_forecaster.py` | The GNN–Pearson model: graph construction, GCN, regime-aware SAAR loss, and walk-forward evaluation |
| `notebooks/V72_Final.ipynb` | GNN training and walk-forward evaluation |
| `notebooks/*_Baseline_GDP_Forecast.ipynb` | Baseline models (ARIMAX, XGBoost, LSTM, Graph Transformer, SPF) |
| `notebooks/Figure*.ipynb` | Figure generators (feature graph, architecture) |

## Method

- **Graph:** 40 macroeconomic-indicator nodes connected by thresholded Pearson correlation.
- **Node features:** 6-month lookback × 4 channels (level, change, volatility, acceleration) = 24 dims.
- **Model:** 3-layer GCN with residual connections and drop-edge regularization, GDP-node readout plus an MLP head.
- **Loss:** a regime-aware SAAR loss with a directional penalty, up-weighted at sign-change quarters.
- **Target:** final-revised nominal GDP, seasonally adjusted annualized rate (SAAR), evaluated by mean absolute error in percentage points.

## Data

Input data (FRED monthly series, MRTS retail sales, BEA GDP vintages) is **not** included in
this repository. Series identifiers are documented in the notebooks; FRED and BEA data are
publicly available, and MRTS files can be downloaded from the U.S. Census Bureau. The
notebooks expect data under a local `./data` directory.

## Requirements

Python 3.12 with `torch`, `torch_geometric`, `xgboost`, `pandas`, `numpy`, `scipy`,
`scikit-learn`, `statsmodels`, and `matplotlib`.
