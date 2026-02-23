from pathlib import Path
import pandas as pd
import pandas_datareader.data as web

DATA_DIR = Path("1.data_set")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "rates_merged.csv"

FRED_SERIES = [
    "DGS1MO", "DGS3MO", "DGS6MO",
    "DGS1", "DGS2", "DGS3", "DGS5",
    "DGS7", "DGS10", "DGS20", "DGS30",
    "SOFR", "FEDFUNDS",
]

START_DATE = None
END_DATE = None


def main():
    dfs = []
    for sid in FRED_SERIES:
        df = web.DataReader(sid, "fred")
        if START_DATE is not None or END_DATE is not None:
            df = df.loc[START_DATE:END_DATE]
        dfs.append(df)

    merged = pd.concat(dfs, axis=1, join="outer").sort_index()
    merged.to_csv(OUT_FILE)

    print(f"saved: {OUT_FILE}")


if __name__ == "__main__":
    main()