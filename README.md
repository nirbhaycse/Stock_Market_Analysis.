# Stock Market Analysis and Forecasting

A Python project that explores historical stock prices and time-series forecasting through an interactive Streamlit application. The current application uses **Prophet** to forecast closing prices for selected companies and visualizes historical data and forecast components.

**Live Demo:** https://sma-stock-market-analysis.vercel.app/  
**Repository:** https://github.com/nirbhaycse/Stock_Market_Analysis.

## Project Overview

The application currently supports:

- Selecting Google (GOOG), Apple (AAPL), Microsoft (MSFT), or GameStop (GME).
- Downloading historical prices with `yfinance`.
- Viewing recent raw price data.
- Plotting historical opening and closing prices with Plotly.
- Fitting a Prophet model to the historical closing-price series.
- Generating and visualizing a forecast for a user-selected horizon of 1–4 years.
- Viewing Prophet forecast components.

The main app is in `Stock_Prediction_App.py`; the notebook version is `Stock_Prediction_App.ipynb`.

## Tech Stack

- **Python**
- **Streamlit** — interactive application interface
- **yfinance** — historical market data
- **Prophet** — time-series forecasting
- **Plotly** — interactive charts
- **Pandas** — data preparation
- **scikit-learn** — metrics in the optional evaluation script

## Methodology

1. **Data collection:** download historical stock data from Yahoo Finance using `yfinance`.
2. **Exploration:** display recent observations and plot opening and closing prices.
3. **Forecasting:** fit Prophet on the `Date` and `Close` columns, renamed to Prophet's required `ds` and `y` fields.
4. **Visualization:** show the forecast and Prophet's trend/seasonality components.
5. **Optional holdout evaluation:** use `evaluate_forecast.py` to compare Prophet with a previous-close baseline on the latest chronological portion of the data.

**Scope clarification:** the checked-in Streamlit application uses Prophet. LSTM, CNN, hybrid LSTM-CNN, and Min-Max scaling are not implemented in that application file. They should only be described as implemented project components if another verified notebook or source file contains those experiments.

## Run the Streamlit App Locally

### Prerequisites

- Python 3.9 or a compatible Python version supported by the installed dependencies
- pip

### 1. Clone the repository

```bash
git clone "https://github.com/nirbhaycse/Stock_Market_Analysis..git"
cd "Stock_Market_Analysis."
```

The repository name ends with a period, so keep the quoted URL/path as shown.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install app dependencies

```bash
python -m pip install streamlit yfinance prophet plotly pandas
```

### 4. Launch the app

```bash
streamlit run Stock_Prediction_App.py
```

Streamlit will print the local URL to open in your browser.

## Evaluate Forecasts on a Chronological Holdout

An additional script, `evaluate_forecast.py`, evaluates Prophet on the latest portion of the historical data and compares it with a simple previous-close baseline. The test dates are taken from actual downloaded observations, rather than generated calendar days.

Install the evaluation dependencies:

```bash
python -m pip install yfinance prophet pandas scikit-learn
```

Run the default Apple example:

```bash
python evaluate_forecast.py --ticker AAPL --start 2016-01-01
```

Try another supported ticker:

```bash
python evaluate_forecast.py --ticker MSFT --start 2016-01-01 --test-fraction 0.2
```

The script reports MAE, RMSE, and MAPE for Prophet and the baseline on the same holdout period. **No performance figures are stated here because the script has not been executed in this environment.** Run it and record the output, dependency versions, and evaluation dates before reporting numerical results.

## Limitations

- Historical price patterns do not guarantee future outcomes.
- A forecast chart alone does not demonstrate accuracy.
- Results depend on the chosen ticker, data period, and evaluation method.
- The optional holdout evaluation is a basic starting point; more rigorous work could use rolling-origin validation, additional baselines, and careful analysis of data leakage.
- The app uses externally retrieved data, so results may change when it is run again.

## Future Improvements

- Add a pinned dependency file such as `requirements.txt`.
- Add automated tests and error handling for unavailable data.
- Compare Prophet with naïve, moving-average, and other appropriate baselines.
- Use rolling or walk-forward evaluation across multiple time periods.
- Document the exact dataset period, model configuration, and measured results.
- Add screenshots and a reproducible results table after the experiments have been run.

## Academic Context

This project was developed as a B.Tech final-year team project in Computer Science and Engineering at Feroze Gandhi Institute of Engineering and Technology, Raebareli, affiliated with Dr. A.P.J. Abdul Kalam Technical University (AKTU), under faculty supervision.

## Disclaimer

This repository is for educational purposes only. It is not investment advice, and model outputs should not be used as the sole basis for financial decisions.
