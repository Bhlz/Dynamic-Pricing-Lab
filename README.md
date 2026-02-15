# ⚡ Dynamic Pricing Lab

**Real-time pricing optimization with demand uncertainty, Monte Carlo belief distributions, and Thompson Sampling.**

A fully functional demo that simulates realistic e-commerce demand, trains gradient-boosted demand models with uncertainty quantiles, optimizes price tick-by-tick, and visualizes *how the system thinks* through 6 interactive dashboard views.

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly)

---

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# Or use Make:
make install
make run
```

The app opens at **http://localhost:8501**.

---

##  Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit App (app.py)              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│  │ Live │ │Think │ │Score │ │Belief│ │Update│ │Base  │
│  │      │ │      │ │board │ │      │ │ Step │ │lines │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
├─────────────────────────────────────────────────────┤
│  Optimizer  ◄──  Monte Carlo  ◄──  Demand Model     │
│  (optimizer.py)  (montecarlo.py)    (model.py)       │
├─────────────────────────────────────────────────────┤
│  Feature Store   ◄──  Simulator ("Real World")      │
│  (features.py)       (simulator.py)                  │
├─────────────────────────────────────────────────────┤
│                Config (config.py)                    │
└─────────────────────────────────────────────────────┘
```

---

##  Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| ** Live** | KPI cards (price, revenue/profit, stock, traffic) + price & objective time series with shock markers |
| ** Think** | Demand vs price curve with p10–p90 uncertainty band + objective curve with vertical lines at current & recommended price |
| ** Scoreboard** | Bar chart of all 101 candidate prices vs expected objective, highlighted chosen price with rich tooltips |
| ** Belief** | Monte Carlo belief distribution — P(each price is optimal) — drives Thompson Sampling exploration |
| ** Update** | Prediction (p10/mean/p90) vs actual demand, rolling error chart, drift detection alerts |
| ** Baselines** | Cumulative optimizer vs Fixed-$100 vs Competitor−$1 baselines with uplift % |

---

##  Controls

| Control | Options |
|---------|---------|
| **Objective** | Revenue / Profit |
| **Exploration** | ON (Thompson Sampling) / OFF (Greedy) |
| **Speed** | 1x / 5x / 20x |
| **Unit Cost** | $30–$120 slider |
| **Max Δ Price/Tick** | 0.5–10 slider |
| **Initial Stock** | 50–1000 slider |
| **Competition** | ON / OFF toggle |

---

##  How It Works

1. **Historical data** (180 days × hourly) is generated with realistic confounding (high traffic → higher price, promos → lower price)
2. **3 Gradient Boosted models** are trained: mean demand, quantile p10, quantile p90
3. **Each live tick** (1 simulated minute):
   - World updates traffic, competition, promos, stock
   - Optimizer evaluates 101 candidate prices with guardrails
   - Monte Carlo samples demand (Negative Binomial) to build belief distribution
   - Price is chosen (greedy or Thompson Sampling)
   - Model retrains every 30 ticks on rolling window
4. **Shocks** hit at tick 80 (traffic −35%), 160 (competitor −12%), 240 (promo for 30 ticks)

---

##  File Structure

| File | Purpose |
|------|---------|
| `config.py` | All tunable parameters (dataclass) |
| `simulator.py` | Demand generation (NegBin) + live world |
| `features.py` | Feature engineering + rolling store |
| `model.py` | GBR mean + quantile models |
| `montecarlo.py` | Belief distribution via sampling |
| `optimizer.py` | Grid optimization + Thompson Sampling |
| `app.py` | Streamlit UI (6 tabs) |

---

##  Notes

- **No real data** — everything is simulated but follows realistic demand patterns
- Uses **scikit-learn** only (no LightGBM dependency issues)
- Demand follows **Negative Binomial** distribution (overdispersed counts)
- Historical data includes **confounding** to simulate real-world bias
# Dynamic-Pricing-Lab
