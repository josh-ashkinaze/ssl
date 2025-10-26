from dataclasses import dataclass
import arviz as az
import aesara.tensor as at
import pandas as pd


@dataclass
class ColumnMap:
    edge_type: str = "edge_type"
    transmitted: str = "tx"
    updated: str = "update"
    delta_b: str = "delta_b"
    b_listener_pre: str = "b_listener_pre"
    b_source: str = "b_source"
    model_id: str = "model_id"  # optional for HA

@dataclass
class Posteriors:
    tx_HH: az.InferenceData
    tx_AH: az.InferenceData
    upd_HH: az.InferenceData
    upd_AH: az.InferenceData
    step_geom_lambda_HH: az.InferenceData
    step_geom_lambda_AH: az.InferenceData

def load_study2_csv(path: str,
                    colmap: ColumnMap = ColumnMap()) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Check for missing values
    required = [colmap.edge_type, colmap.transmitted, colmap.updated,
                colmap.delta_b, colmap.b_listener_pre, colmap.b_source]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df.copy()
    df[colmap.edge_type] = df[colmap.edge_type].astype(str)

    for bcol in [colmap.transmitted, colmap.updated]:
        df[bcol] = df[bcol].astype(int)

    for bcol in [colmap.delta_b, colmap.b_listener_pre, colmap.b_source]:
        df[bcol] = df[bcol].astype(int)

    return df