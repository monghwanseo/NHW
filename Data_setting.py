from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path("1.data_set")
DATA_FILE = DATA_DIR / "rates_merged.csv"


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.sort_values("DATE").reset_index(drop=True)


def prepare_short_rate(df):
    df = df.copy()
    df["r"] = df["SOFR"] / 100.0
    df = df.dropna(subset=["r"]).reset_index(drop=True)
    df[["DATE", "r"]].to_csv(DATA_DIR / "short_rate.csv", index=False)
    return df


def build_pairs(df):
    df = df.copy()
    df["r_next"] = df["r"].shift(-1)
    pair_df = df.dropna(subset=["r", "r_next"]).reset_index(drop=True)
    pair_df[["DATE", "r", "r_next"]].to_csv(DATA_DIR / "short_rate_pairs.csv", index=False)
    return pair_df


def estimate_vasicek_params(pair_df, dt=1 / 252):
    r_t = pair_df["r"].to_numpy(dtype=float)
    r_next = pair_df["r_next"].to_numpy(dtype=float)

    X = np.column_stack([np.ones_like(r_t), r_t])
    beta = np.linalg.inv(X.T @ X) @ (X.T @ r_next)
    c_hat, phi_hat = float(beta[0]), float(beta[1])

    eps = r_next - (c_hat + phi_hat * r_t)
    sigma_eps2_hat = float(np.mean(eps ** 2))

    a_hat = float(-np.log(phi_hat) / dt)
    theta_hat = float(c_hat / (1.0 - phi_hat))
    sigma_hat = float(np.sqrt(sigma_eps2_hat * 2.0 * a_hat / (1.0 - phi_hat ** 2)))

    params = {
        "a_hat": a_hat,
        "theta_hat": theta_hat,
        "sigma_hat": sigma_hat,
        "phi_hat": phi_hat,
        "c_hat": c_hat,
        "sigma_eps2_hat": sigma_eps2_hat,
        "dt": float(dt),
    }

    with open(DATA_DIR / "vasicek_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)

    return params


def main():
    df = load_data()
    df = prepare_short_rate(df)
    pair_df = build_pairs(df)
    estimate_vasicek_params(pair_df)

    print("done")


if __name__ == "__main__":
    main()