import yfinance as yf

tickers = ["^NDX", "DX-Y.NYB"]

data = yf.download(
    tickers,
    start="2016-01-01",
    end="2026-07-01",
    interval="1d",
    auto_adjust=True
)

if hasattr(data.columns, "levels"):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)

data.to_csv(
    "data/Historical/Financial_data.csv",
    index=False
)

print(data)
print(f"\nRows: {len(data)}")
print(f"Start: {data['Date'].min()}")
print(f"End: {data['Date'].max()}")