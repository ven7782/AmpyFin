﻿
# 🌟 AmpyFin Trading Bot

## 🚀 Introduction

Welcome to **AmpyFin**, an advanced AI-powered trading bot designed for the NASDAQ-100. Imagine having expert traders working for you 24/7—AmpyFin makes this a reality.

## 📊 AmpyFin’s Data Collection Power

### 🔍 Data Sources

- **Financial Modeling Prep API**: Retrieves NASDAQ-100 tickers to gain crucial market insights.
- **Polygon API**: Monitors real-time market conditions, ensuring that the bot acts based on the most current data.

### 💾 Data Storage

All data and trading logs are securely stored in **MongoDB**, allowing fast access to historical trading information and supporting in-depth analysis.

### 🤖 Machine Learning at Work

At the core of AmpyFin are diverse algorithms optimized for different market conditions. Rather than relying on a single strategy or multiple strategies, AmpyFin relies on a ranked ensemble learning system that dynamically ranks each strategy and gives more influence in the final decision to strategies with better performance.

### 📈 Trading Strategies

Some of the strategies AmpyFin employs include:

- **📊 Mean Reversion**: Predicts asset prices will return to their historical average.
- **📈 Momentum**: Capitalizes on prevailing market trends.
- **💱 Arbitrage**: Identifies and exploits price discrepancies between related assets.
- **🧠 AI-Driven Custom Strategies**: Continuously refined through machine learning for enhanced performance.

These strategies work collaboratively, ensuring AmpyFin is always prepared for changing market dynamics.

### 🔗 How Dynamic Ranking Works

Managing multiple algorithms is simplified with AmpyFin’s dynamic ranking system, which ranks each algorithm based on performance.

#### 🏆 Ranking System

Each strategy starts with a base score of 0 and a mock balance of $50,000. The system evaluates their performance and assigns a weight based on the following function:

$$
\left( \frac{e^e}{e^2} - 1 \right)^{2i}
$$

Where \(i\) is the strategy's rank. Please keep in mind that the strategy's rank 

#### ⏳ Time Delta Coefficient

This ensures that strategies with better recent performance have a greater influence on decision-making while maintaining balance by also accounting for old performance as well.

### 💡 Benefits of Dynamic Ranking

- **📉 Quickly adapts to changing market conditions.**
- **📊 Prioritizes high-performing algorithms.**
- **⚖️ Balances risk while maximizing potential returns.**

## 📂 File Structure and Objectives


### 🤝 trading_client.py

**Objective**: Executes trades based on algorithmic decisions.

**Features**:

- Executes trades every 60 seconds by default (adjustable based on user).
- Ensures a minimum spending balance of $15,000 (adjustable based on user) and maintains 30% liquidity (adjustable based on user).
- Logs trades with details like timestamp, stock, and reasoning.

### 🏆 ranking_client.py

**Objective**: Runs the ranking system to evaluate trading strategies.

**Features**:

- Downloads NASDAQ-100 tickers and stores them in MongoDB.
- Updates algorithm scores and rankings every 30 seconds (adjustable based on user).

### 📜 strategies/*

**Objective**: Defines various trading strategies. Houses strategies like mean reversion, momentum, and arbitrage.

**Features**:

- **trading_strategies_v1.py**: Archived first iteration of AmpyFin used 5 strategies. This file is not supported anymore but is a great reference material
- **trading_strategies_v2.py**:  Archived second gen older strategies being used in the ranking system. Contains 50 strategies with a lot leaning towards momentum.
- **trading_strategies_v2_1.py**: Archived second gen older strategies that complements the older strategies in trading_strategies_v2.py. Houses 10 more strategies. This is where newer strategies will be implemented until it caps at 50 strategies as well.
- **talib_indicators.py**: Contains all the technical indicators used in the strategies. To visit the documentation for each technical indicator, please visit the following link: [Link to TA](https://ta-lib.org/). These indicators were not developed by me, but I have modified their use to fit the needs of AmpyFin.

### 🔧 helper_files/*

**Objective**: Helper Files to help with both trading client and ranking client. Houses functions for retrieving a Mongo Client, getting latest prices, current strategies implemented etc.

**Features**:

- **client_helper.py**: Contains common functions for client operations in both ranking and trading.


## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yeonholee50/AmpyFin.git
cd AmpyFin
```

### 2️⃣ Install Dependencies

- Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```

- We have recently migrated to using Ta-Lib for trading. Please follow the installation instructions here: 

👉 [Ta-Lib Python Original](https://github.com/TA-Lib/ta-lib-python)

👉 [Ta-Lib Python Easy Installation](https://github.com/cgohlke/talib-build/releases)

### 3️⃣ Configuration

1. **Create `config.py`**:
   - Copy `config_template.py` to `config.py` and enter your API keys and MongoDB credentials.
    ```python
    POLYGON_API_KEY = "your_polygon_api_key"
    FINANCIAL_PREP_API_KEY = "your_fmp_api_key"
    MONGO_DB_USER = "your_mongo_user"
    MONGO_DB_PASS = "your_mongo_password"
    API_KEY = "your_alpaca_api_key"
    API_SECRET = "your_alpaca_secret_key"
    BASE_URL = "https://paper-api.alpaca.markets"
    mongo_url = "your mongo connection string"
    ```

### 4️⃣ API Setup

- Polygon API
1. Sign up at [Polygon.io](https://polygon.io/) and get an API key.
2. Add it to `config.py` as `POLYGON_API_KEY`.

- Financial Modeling Prep API
1. Sign up at [Financial Modeling Prep](https://financialmodelingprep.com/) and get an API key.
2. Add it to `config.py` as `FINANCIAL_PREP_API_KEY`.

- Alpaca API
1. Sign up at [Alpaca](https://alpaca.markets/) and get API keys.
2. Add them to `config.py` as `API_KEY` and `API_SECRET`.

### 5️⃣ Set Up MongoDB

- Sign up for a MongoDB cluster (e.g., via MongoDB Atlas).
- Create a database for stock data storage and replace the `mongo_url` in 'config.py' with your connection string. Make sure to give yourself Network Access.
- Run the setup script `setup.py`:
- After running the mongo setup script, the MongoDB setup for the rest will be completed on the first minute in trading for both ranking and trading.


## ⚡ Usage

To run the bot, execute on two separate terminals:

```bash
python ranking_client.py
python trading_client.py
```

## ⚠️ IMPORTANT

For people looking to do live trading, I suggest training via running ranking_client.py for at least two weeks before running the trading bot altogether. This way, you're running with a client that has been trained to a certain extent (with strategies ranked) and is ready to go. Otherwise, you will most likely be buying random stocks.

## 📑 Logging

- **system.log**: Tracks major events like API errors and MongoDB operations.
- **rank_system.log**: Logs all ranking-related events and updates.

## 🛠️ Contributing

Contributions are welcome! 🎉 Feel free to submit pull requests or report issues. All contributions should be made on the **test branch**. Please avoid committing directly to the **main branch**.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.
