# Stock Market Analysis and Prediction

A machine learning project focused on analysing historical stock market data and exploring time-series forecasting techniques using Python.

This project was developed as part of my B.Tech final-year project in Computer Science and Engineering, under faculty supervision and in collaboration with my project team. It introduced me to financial time-series data, data preprocessing, and the use of machine learning for prediction problems.

## Overview

Stock prices change over time and are influenced by many factors. Analysing historical data can help us understand price patterns and investigate how forecasting models behave on time-dependent datasets.

The objective of this project was to study historical stock market data, prepare it for machine learning, and explore deep learning approaches for stock price prediction.

**Live Demo:** [Stock Market Analysis](https://sma-stock-market-analysis.vercel.app/)

**Source Code:** [GitHub Repository](https://github.com/nirbhaycse/Stock_Market_Analysis)

## Objectives

- Retrieve and analyse historical stock market data.
- Understand the preprocessing steps required before model training.
- Apply normalization to numerical data using Min-Max scaling.
- Explore Long Short-Term Memory (LSTM) networks for time-series forecasting.
- Study different machine learning approaches to understand their potential for stock price prediction.
- Understand the challenges involved in forecasting financial data.

## Technologies Used

- **Python** — data processing and model development
- **yfinance** — retrieving historical financial market data
- **Pandas and NumPy** — data manipulation and numerical operations
- **Jupyter Notebook** — experimentation and analysis
- **Scikit-learn** — preprocessing and machine learning utilities
- **TensorFlow/Keras** — deep learning, where used in the project implementation

*Note: The libraries listed above should be checked against the current source code before final submission.*

## Methodology

The project follows a general machine learning workflow:

### 1. Data Collection

Historical stock market data is retrieved using `yfinance`. The data provides a basis for examining price movements over time.

### 2. Data Preprocessing

The collected data is prepared for model development. The project includes Min-Max normalization and the division of data into training and testing sets.

For time-series prediction, maintaining the chronological order of observations is important because future observations should not be used to train a model evaluated on earlier data.

### 3. Model Development

Long Short-Term Memory (LSTM) networks were studied for forecasting stock prices. LSTMs are a type of recurrent neural network designed to learn patterns in sequential data.

The project report also discusses CNN and hybrid LSTM-CNN approaches, along with traditional machine learning methods. The implementation of each method should be verified in the source notebooks before treating it as an implemented and evaluated model.

### 4. Prediction and Analysis

The prepared data is used to explore stock price forecasting. The predictions should be interpreted carefully because financial time series are noisy, non-stationary, and difficult to predict reliably.

A meaningful evaluation should compare predictions with actual values on unseen, chronologically later data.

## Key Learning Outcomes

Through this project, I gained experience with:

- Working with historical financial data using Python.
- Understanding data preprocessing and normalization.
- Learning the fundamentals of LSTM-based time-series forecasting.
- Understanding the importance of training and testing datasets.
- Recognizing the challenges of evaluating machine learning models on financial data.
- Working on a collaborative academic project.

The project also motivated me to learn more about machine learning methods, experimental evaluation, and the reliability of predictions on real-world datasets.

## Limitations and Future Improvements

Stock price forecasting is a challenging problem, and historical patterns do not guarantee future performance. This project should be treated as an academic learning exercise, not as a reliable trading system or financial advice.

Potential improvements include:

- Establishing simple baseline models before evaluating deep learning methods.
- Comparing models using MAE and RMSE on a chronological test set.
- Checking for data leakage during preprocessing and model evaluation.
- Recording experimental settings and results for reproducibility.
- Evaluating performance across different time periods and stocks.
- Documenting the limitations of each forecasting approach.

These improvements would make the experimental process more systematic and help assess whether a model generalizes beyond the data used during training.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/nirbhaycse/Stock_Market_Analysis.git
   cd Stock_Market_Analysis
   ```

2. Create and activate a Python virtual environment.

   ```bash
   python -m venv .venv
   ```

   On Windows:

   ```bash
   .venv\Scripts\activate
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies required by the project:

   ```bash
   pip install pandas numpy matplotlib scikit-learn yfinance jupyter
   ```

   Install TensorFlow only if it is required by the notebooks:

   ```bash
   pip install tensorflow
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open the relevant notebook in the repository and execute its cells in order.

   If a notebook requires additional libraries or configuration, install or configure them as described in that notebook.

## Academic Context

**Project:** Stock Market Analysis  
**Degree:** B.Tech in Computer Science and Engineering  
**Institution:** Feroze Gandhi Institute of Engineering and Technology, Raebareli  
**Affiliation:** Dr. A.P.J. Abdul Kalam Technical University (AKTU)

This was a team project completed under faculty supervision.

## Disclaimer

This project is for educational and experimental purposes only. It does not provide investment recommendations, and its predictions should not be used as the sole basis for financial decisions.
