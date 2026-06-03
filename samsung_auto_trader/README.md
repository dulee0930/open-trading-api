# Samsung Auto Trader

A modular Python auto-trading script for Samsung Electronics (`005930`) using the Korea Investment Open API mock trading REST endpoints.

## Features

- OAuth token reuse via local cache
- REST-only trading loop
- Buy at `current_price - 2000`, sell at `current_price + 2000`
- Execution verification through account holdings and cash updates
- Active only during trading hours `09:10` to `15:30`
- Configurable via environment variables

## Files

- `config.py` - loads settings and environment variables
- `auth.py` - manages token caching and refresh
- `api_client.py` - REST request wrapper with retry/401 handling
- `market_data.py` - fetches current price for `005930`
- `account.py` - retrieves account summary and holdings
- `orders.py` - places buy/sell stock orders
- `trader.py` - main trading loop and order execution logic
- `main.py` - entrypoint to start the trader

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment

Set credentials via environment variables:

- `GH_ACCOUNT` or `ACCOUNT_NUMBER` (required)
- `GH_APPKEY` (required)
- `GH_APPSECRET` (required)
- `GH_PRODUCT_CODE` or `PRODUCT_CODE` (optional, default `01`)

Example:

```bash
export GH_ACCOUNT="your_account"
export GH_APPKEY="your_appkey"
export GH_APPSECRET="your_appsecret"
export PRODUCT_CODE="01"
```

## Run

From inside the `samsung_auto_trader` directory:

```bash
python main.py
```

From the repository root:

```bash
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/main.py
```

## Validation

- Verified token authentication successfully using stored env vars.
- Confirmed current price lookup for Samsung Electronics (`005930`).
- Confirmed holdings lookup and available cash extraction from the Korea Investment mock API.
- Trading loop is time-zone-aware and uses Asia/Seoul market hours.

## Historical Price Export

You can export Samsung Electronics historical price data to CSV using the added export script.

From the repository root:

```bash
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/export_history.py \
  --symbol 005930 \
  --output samsung_price_history.csv \
  --period D \
  --adj 1 \
  --market J
```

Options:

- `--symbol`: stock code to export (default `005930`)
- `--output`: CSV output path
- `--period`: `D` for daily, `W` for weekly, `M` for monthly
- `--adj`: `1` for adjusted prices, `0` for unadjusted
- `--market`: `J` for KRX, `NX` for NXT, `UN` for unified

## Notes

- The bot is designed for mock/demo trading using the Korea Investment Open API.
- It polls the market and account state periodically, and it only runs during preconfigured trading hours.
- Confirm API field names and mock endpoint availability before using in production.
