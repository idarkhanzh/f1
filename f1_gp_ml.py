# Robust F1 GP predictor (FastF1 + scikit-learn)
# - Creates cache dir automatically
# - Uses DriverAbbr (3-letter code) as the join key across FP2/Quali/Race
# - Handles missing columns gracefully
# - Cross-validates by season, then predicts target year using FP2+Quali
#
# Change EVENT_NAME to "Hungary" / "Belgium" / etc.

import os
import warnings
warnings.filterwarnings("ignore")

import fastf1
from fastf1.core import Laps
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

# -------------------
# Config
# -------------------
CACHE_DIR = "./fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

EVENT_NAME = "Monza"                 # e.g., "Hungary", "Belgium"
TRAIN_YEARS = list(range(2018, 2024 + 1))
PREDICT_YEAR = 2025
RANDOM_STATE = 42

# -------------------
# Helpers
# -------------------
def _time_to_seconds(t):
    """Convert FastF1 time (Timedelta/str/None) to float seconds."""
    if pd.isna(t):
        return np.nan
    if isinstance(t, (pd.Timedelta, np.timedelta64)):
        return pd.to_timedelta(t).total_seconds()
    try:
        return pd.to_timedelta(str(t)).total_seconds()
    except Exception:
        return np.nan

def _safe_col(df, name, default=np.nan):
    """Ensure a column exists; if not, create with default."""
    if name not in df.columns:
        df[name] = default
    return df

def get_event(year, name=EVENT_NAME):
    return fastf1.get_event(year, name)

def load_session(year, session_name):
    ev = get_event(year)
    ses = ev.get_session(session_name)    # "FP2", "Q", "R"
    ses.load(laps=True, telemetry=False, weather=False)
    return ses

# ---------- Feature builders ----------
def fp2_features(fp2_session):
    """Median clean-lap pace + lap count per driver from FP2.
       Returns: DriverAbbr, FP2MedianLapSec, FP2LapCount"""
    laps: Laps = fp2_session.laps
    if laps is None or laps.empty:
        return pd.DataFrame(columns=["DriverAbbr", "FP2MedianLapSec", "FP2LapCount"])

    laps = laps.pick_quicklaps()
    if laps.empty:
        return pd.DataFrame(columns=["DriverAbbr", "FP2MedianLapSec", "FP2LapCount"])

    df = laps[["Driver", "LapTime"]].copy()
    df["LapSec"] = df["LapTime"].apply(_time_to_seconds)
    agg = df.groupby("Driver", as_index=False).agg(
        FP2MedianLapSec=("LapSec", "median"),
        FP2LapCount=("LapSec", "count")
    )
    # Standardize join key
    agg = agg.rename(columns={"Driver": "DriverAbbr"})
    return agg

def quali_features(quali_session):
    """Best quali time (min of Q1/Q2/Q3), team, quali position.
       Returns: DriverAbbr, DriverName, Team, BestQSec, QualiPosition"""
    res = quali_session.results
    if res is None or res.empty:
        return pd.DataFrame(columns=["DriverAbbr", "DriverName", "Team", "BestQSec", "QualiPosition"])

    df = res.copy()

    # Normalize common columns
    rename_map = {}
    if "Abbreviation" in df.columns:
        rename_map["Abbreviation"] = "DriverAbbr"
    if "TeamName" in df.columns:
        rename_map["TeamName"] = "Team"
    if "Position" in df.columns:
        rename_map["Position"] = "QualiPosition"
    # Try to build a readable name
    if "FullName" in df.columns:
        rename_map["FullName"] = "DriverName"
    elif "Driver" in df.columns:
        # sometimes a name-like field exists
        rename_map["Driver"] = "DriverName"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    df = _safe_col(df, "DriverAbbr")
    df = _safe_col(df, "DriverName", default=df.get("DriverAbbr", pd.Series(dtype=str)))
    df = _safe_col(df, "Team", default="Unknown")
    df = _safe_col(df, "QualiPosition", default=np.nan)

    # Build BestQSec
    for col in ["Q1", "Q2", "Q3"]:
        if col in df.columns:
            df[col] = df[col].apply(_time_to_seconds)
        else:
            df[col] = np.nan
    df["BestQSec"] = df[["Q1", "Q2", "Q3"]].min(axis=1, skipna=True)

    return df[["DriverAbbr", "DriverName", "Team", "BestQSec", "QualiPosition"]]

def race_target(race_session):
    """Finishing position + starting grid, DNFs sent to back of grid.
       Returns: DriverAbbr, DriverName, Team, Grid, FinishPos"""
    res = race_session.results
    if res is None or res.empty:
        return pd.DataFrame(columns=["DriverAbbr", "DriverName", "Team", "Grid", "FinishPos"])

    df = res.copy()

    rename_map = {}
    if "Abbreviation" in df.columns:
        rename_map["Abbreviation"] = "DriverAbbr"
    if "TeamName" in df.columns:
        rename_map["TeamName"] = "Team"
    if "FullName" in df.columns:
        rename_map["FullName"] = "DriverName"
    elif "Driver" in df.columns:
        rename_map["Driver"] = "DriverName"

    if rename_map:
        df = df.rename(columns=rename_map)

    df = _safe_col(df, "DriverAbbr")
    df = _safe_col(df, "DriverName", default=df.get("DriverAbbr", pd.Series(dtype=str)))
    df = _safe_col(df, "Team", default="Unknown")

    # Finish position + DNF handling
    df["FinishPos"] = pd.to_numeric(df.get("Position"), errors="coerce")
    back_of_grid = max(20, int(df["FinishPos"].max(skipna=True) or 20))
    status = df.get("Status")
    if status is not None:
        dnf_mask = status.astype(str).str.contains("DNF|DSQ|DNS|DNQ|RET", case=False, na=False)
        df.loc[dnf_mask, "FinishPos"] = back_of_grid
    df["FinishPos"] = df["FinishPos"].fillna(back_of_grid).astype(int)

    # Grid position
    if "GridPosition" in df.columns:
        df["Grid"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    else:
        df["Grid"] = np.nan

    return df[["DriverAbbr", "DriverName", "Team", "Grid", "FinishPos"]]

# ---------- Dataset ----------
def build_dataset(years):
    rows = []
    for y in tqdm(years, desc="Building dataset"):
        try:
            fp2 = load_session(y, "FP2")
            q = load_session(y, "Q")
            r = load_session(y, "R")
        except Exception as e:
            print(f"[WARN] {y}: failed to load sessions: {e}")
            continue

        df_fp2 = fp2_features(fp2)          # DriverAbbr, FP2MedianLapSec, FP2LapCount
        df_q   = quali_features(q)          # DriverAbbr, DriverName, Team, BestQSec, QualiPosition
        df_r   = race_target(r)             # DriverAbbr, DriverName, Team, Grid, FinishPos

        # Merge on DriverAbbr only (team names can vary in formatting)
        df = df_q.merge(df_fp2, on="DriverAbbr", how="left") \
                 .merge(df_r,  on="DriverAbbr", how="right", suffixes=("", "_R"))

        # Prefer race's DriverName/Team if present
        df["DriverName"] = df["DriverName_R"].fillna(df["DriverName"])
        df["Team"] = df["Team_R"].fillna(df["Team"])

        # Fallbacks
        df["Grid"] = df["Grid"].fillna(pd.to_numeric(df["QualiPosition"], errors="coerce"))
        if "BestQSec" not in df or df["BestQSec"].isna().all():
            df["BestQSec"] = np.nan
        df["BestQSec"] = df["BestQSec"].fillna(df["BestQSec"].max(skipna=True) or 200.0)

        df["Year"] = y
        df = df.dropna(subset=["FinishPos"])

        rows.append(df[[
            "DriverAbbr", "DriverName", "Team", "Year",
            "FP2MedianLapSec", "FP2LapCount", "BestQSec", "Grid", "FinishPos"
        ]])

    if not rows:
        return pd.DataFrame(columns=[
            "DriverAbbr", "DriverName", "Team", "Year",
            "FP2MedianLapSec", "FP2LapCount", "BestQSec", "Grid", "FinishPos"
        ])

    data = pd.concat(rows, ignore_index=True)

    # Cleanups
    data["FP2MedianLapSec"] = data["FP2MedianLapSec"].fillna(data["FP2MedianLapSec"].median())
    data["FP2LapCount"]     = data["FP2LapCount"].fillna(0)
    data["Grid"]            = data["Grid"].fillna(data["Grid"].median())
    data["BestQSec"]        = data["BestQSec"].fillna(data["BestQSec"].median())

    data["Team"]        = data["Team"].astype("category")
    data["DriverAbbr"]  = data["DriverAbbr"].astype("category")
    data["Year"]        = data["Year"].astype(int)
    data["FinishPos"]   = data["FinishPos"].astype(int)

    # If DriverName missing, fall back to abbr
    if "DriverName" not in data or data["DriverName"].isna().all():
        data["DriverName"] = data["DriverAbbr"].astype(str)

    return data

# ---------- Model ----------
def make_model():
    cat_features = ["Team", "DriverAbbr"]
    num_features = ["FP2MedianLapSec", "FP2LapCount", "BestQSec", "Grid", "Year"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline([("pre", pre), ("rf", model)])

def evaluate_cv(df):
    """Group K-Fold by Year to reduce leakage across seasons."""
    X = df.drop(columns=["FinishPos", "DriverName"])
    y = df["FinishPos"].values
    groups = df["Year"].values

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    maes = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_model()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
    return float(np.mean(maes)), float(np.std(maes))

def train_final(df):
    X = df.drop(columns=["FinishPos", "DriverName"])
    y = df["FinishPos"].values
    pipe = make_model()
    pipe.fit(X, y)
    return pipe

def build_predict_features(year):
    """Assemble features for the target year (requires FP2 + Quali)."""
    try:
        fp2 = load_session(year, "FP2")
        q   = load_session(year, "Q")
    except Exception as e:
        raise RuntimeError(f"Could not load FP2/Q for {year}: {e}")

    df_fp2 = fp2_features(fp2)   # DriverAbbr, FP2MedianLapSec, FP2LapCount
    df_q   = quali_features(q)   # DriverAbbr, DriverName, Team, BestQSec, QualiPosition

    df = df_q.merge(df_fp2, on="DriverAbbr", how="left")

    # Fallbacks
    df["FP2MedianLapSec"] = df["FP2MedianLapSec"].fillna(df["FP2MedianLapSec"].median())
    df["FP2LapCount"]     = df["FP2LapCount"].fillna(0)
    df["Grid"]            = pd.to_numeric(df["QualiPosition"], errors="coerce").fillna(df["QualiPosition"].median())
    df["BestQSec"]        = df["BestQSec"].fillna(df["BestQSec"].median())
    df["Year"]            = year

    # Keep model features
    features = df[[
        "DriverAbbr", "DriverName", "Team", "Year",
        "FP2MedianLapSec", "FP2LapCount", "BestQSec", "Grid"
    ]].copy()

    features["Team"]       = features["Team"].astype("category")
    features["DriverAbbr"] = features["DriverAbbr"].astype("category")
    # Ensure DriverName exists for pretty output
    if "DriverName" not in features or features["DriverName"].isna().all():
        features["DriverName"] = features["DriverAbbr"].astype(str)

    return features

# ---------- Main ----------
def main():
    # 1) Build historical dataset
    data = build_dataset(TRAIN_YEARS)
    if data.empty:
        raise SystemExit("No training data assembled. Check cache/network and chosen years.")

    print(f"Dataset shape: {data.shape}")
    print(data.head(10).to_string(index=False))

    # 2) Cross-validated evaluation (by year)
    mean_mae, std_mae = evaluate_cv(data)
    print(f"\nYear-grouped CV MAE (finish position): {mean_mae:.2f} ± {std_mae:.2f}")

    # 3) Train final model on all training years
    model = train_final(data)

    # 4) Predict target year's GP (requires FP2 + Quali)
    try:
        X_pred = build_predict_features(PREDICT_YEAR)
        preds = model.predict(X_pred.drop(columns=["DriverName"]))
        X_pred = X_pred.assign(PredictedFinishPos=preds)
        X_pred = X_pred.sort_values("PredictedFinishPos").reset_index(drop=True)
        X_pred["PredictedRank"] = np.arange(1, len(X_pred) + 1)

        cols = ["PredictedRank", "DriverName", "DriverAbbr", "Team", "Grid",
                "BestQSec", "FP2MedianLapSec", "FP2LapCount", "PredictedFinishPos"]
        print("\nPredicted classification (lower is better):")
        print(X_pred[cols].to_string(index=False))
    except RuntimeError as e:
        print(f"\n[INFO] Prediction skipped: {e}")
        print("Tip: run this script after FP2 and Quali are complete for the target year.")

if __name__ == "__main__":
    main()
