# Samsung Auto Trader

This subproject provides a Samsung Electronics (`005930`) auto-trading workflow using the Korea Investment mock trading REST API.

## What it contains

- `main.py` / `trader.py`: trading loop, signal validation, and order execution
- `export_history.py`: refreshes the internal historical CSV used by the signal generator
- `tiny_gpt_trading_signal_real_cli.py`: AI signal generator that writes a buying/selling signal JSON
- `test_trading_logic.py`: local validation for signal-driven order decisions

## Requirements

Install dependencies from the subproject:

```bash
cd samsung_auto_trader
pip install -r requirements.txt
```

## Environment

Required environment variables:

- `GH_ACCOUNT` or `ACCOUNT_NUMBER`
- `GH_APPKEY`
- `GH_APPSECRET`
- `GH_PRODUCT_CODE` or `PRODUCT_CODE` (optional, default `01`)

Example:

```bash
export GH_ACCOUNT="your_account"
export GH_APPKEY="your_appkey"
export GH_APPSECRET="your_appsecret"
export PRODUCT_CODE="01"
```

## Execution flow

1. Refresh historical prices.
2. Generate the latest AI signal JSON.
3. Run the trader to execute orders based on the JSON.

## 1. Refresh historical price data

The default export file is `samsung_auto_trader/Samsung_Daily_Data_yfinance.csv`.

Run from the repository root:

```bash
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/export_history.py \
  --symbol 005930 \
  --period D \
  --adj 1 \
  --market J
```

This command appends new rows to the existing internal CSV and preserves the file format.

## 2. Generate the AI signal

The AI generator now defaults to the internal CSV and writes its outputs to the subproject folder.

From the repository root:

```bash
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/tiny_gpt_trading_signal_real_cli.py
```

To specify explicit paths:

```bash
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/tiny_gpt_trading_signal_real_cli.py \
  --csv samsung_auto_trader/Samsung_Daily_Data_yfinance.csv \
  --output-json samsung_auto_trader/latest_trading_signal.json \
  --output-history samsung_auto_trader/trading_signals_history.csv
```

## 3. Run the trader

From the repository root:

```bash
PYTHONPATH=./samsung_auto_trader python samsung_auto_trader/main.py
```

`main.py` reads `samsung_auto_trader/latest_trading_signal.json` by default.
If the JSON file is missing or invalid, the trader will skip execution.

This trader now runs continuously across trading days:

- executes trading cycles during the configured market window
- waits after market close and runs `export_history.py` at 23:00 local time
- generates updated `latest_trading_signal.json` and `trading_signals_history.csv` for the next trading day
- uses the latest end-of-day price data when regenerating the AI signal at 23:00

## Validation

Run the local signal/order logic validation:

```bash
cd samsung_auto_trader
python test_trading_logic.py
```

## Notes

- `samsung_auto_trader/export_history.py` updates the internal CSV file used by the AI generator.
- `samsung_auto_trader/latest_trading_signal.json` and `samsung_auto_trader/trading_signals_history.csv` are generated outputs.
- `samsung_auto_trader/token_cache.json` is a local auth cache and should not be committed.
