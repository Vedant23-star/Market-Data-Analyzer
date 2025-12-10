# 📈 SMA Crossover Backtesting API

This project deploys a financial backtesting engine as a robust, Dockerized REST API using FastAPI.

## Project Scope

The goal was to implement a Simple Moving Average (SMA) crossover strategy, apply comprehensive performance analytics (CAGR, MDD, Sharpe Ratio), and expose the results via a high-performance, well-documented API.

## Key Features

* **API Framework:** FastAPI for high performance and automatic documentation (Swagger UI).
* **Backtesting Strategy:** SMA Crossover (Fast vs. Slow window).
* **Metrics:** Calculates CAGR, Maximum Drawdown (MDD), and Sharpe Ratio for the strategy vs. a Buy & Hold benchmark.
* **Deployment:** Fully containerized using Docker for portability.

## 🐳 Getting Started with Docker (Recommended)

To run this API, you only need Docker installed and running on your system.

1.  **Navigate:** Open your terminal in the directory containing the `Dockerfile` and `main.py`.
2.  **Build the Image:**
    ```bash
    docker build -t quant-api:latest .
    ```
3.  **Run the Container:** This runs the API in the background on port 8000.
    ```bash
    docker run -d --name quant-api-instance -p 8000:8000 quant-api:latest
    ```
4.  **Access the API:** Open your web browser to view the interactive documentation and run the backtest:
    ```
    http://localhost:8000/docs
    ```

## 💻 Running Locally (Alternative)

If you prefer to run the app locally (without Docker):

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Uvicorn:**
    ```bash
    uvicorn main:app --reload
    ```
---