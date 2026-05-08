import os
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

SPECTRUM_EXCEL = r""
MEASURED_EXCEL = r""
OUTPUT_DIR = r""
ID_COL = "plot_id"
VARIETY_COL = "variety"
STAGE_COL = "growth_stage"
TARGET_COL = "initiation_das"   # 推荐用：播种后第几天
TARGET_IS_DATE = False
SOWING_DATE_COL = None
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER = 40
CV_SPLITS = 10
MIN_SAMPLES_PER_STAGE = 10

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
def normalize_col_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")
def safe_divide(a, b):
    return np.where(np.abs(b) < 1e-10, np.nan, a / b)
def extract_number_from_col(col):
    match = re.search(r"(\d+\.?\d*)", str(col))
    if match:
        return float(match.group(1))
    return None
def find_band_column(df, band_name, target_nm, aliases):
    col_map = {normalize_col_name(c): c for c in df.columns}
    for alias in aliases:
        alias_norm = normalize_col_name(alias)
        if alias_norm in col_map:
            return col_map[alias_norm]
    wavelength_cols = []
    for c in df.columns:
        value = extract_number_from_col(c)
        if value is not None:
            wavelength_cols.append((c, value))
    if len(wavelength_cols) == 0:
        raise ValueError(
            f"找不到 {band_name} 波段。请检查 Excel 是否有 {aliases} 或波长列。"
        )
    nearest_col, nearest_nm = min(
        wavelength_cols,
        key=lambda x: abs(x[1] - target_nm)
    )
    print(f"[INFO] {band_name} 使用列：{nearest_col}，对应波长约 {nearest_nm} nm")
    return nearest_col
def get_band_reflectance(df):
    band_config = {
        "blue": {
            "target_nm": 450,
            "aliases": ["blue", "b", "band_blue", "蓝光"]
        },
        "green": {
            "target_nm": 560,
            "aliases": ["green", "g", "band_green", "绿光"]
        },
        "red": {
            "target_nm": 665,
            "aliases": ["red", "r", "band_red", "红光"]
        },
        "red_edge": {
            "target_nm": 705,
            "aliases": ["red_edge", "rededge", "re", "band_red_edge", "红边"]
        },
        "nir": {
            "target_nm": 800,
            "aliases": ["nir", "near_infrared", "band_nir", "近红外"]
        }
    }
    out = pd.DataFrame(index=df.index)
    for band, cfg in band_config.items():
        col = find_band_column(
            df,
            band_name=band,
            target_nm=cfg["target_nm"],
            aliases=cfg["aliases"]
        )
        out[band] = pd.to_numeric(df[col], errors="coerce")
    return out
def calculate_vegetation_indices(df):
    blue = df["blue"].astype(float)
    green = df["green"].astype(float)
    red = df["red"].astype(float)
    red_edge = df["red_edge"].astype(float)
    nir = df["nir"].astype(float)
    vi = pd.DataFrame(index=df.index)
    # 1. NDVI
    vi["NDVI"] = safe_divide(nir - red, nir + red)
    # 2. GNDVI
    vi["GNDVI"] = safe_divide(nir - green, nir + green)
    # 3. NDRE
    vi["NDRE"] = safe_divide(nir - red_edge, nir + red_edge)
    # 4. CIred-edge
    vi["CIre"] = safe_divide(nir, red_edge) - 1
    # 5. RVI
    vi["RVI"] = safe_divide(nir, red)
    # 6. DVI
    vi["DVI"] = nir - red
    # 7. EVI
    vi["EVI"] = 2.5 * safe_divide(
        nir - red,
        nir + 6 * red - 7.5 * blue + 1
    )
    # 8. SAVI
    vi["SAVI"] = 1.5 * safe_divide(
        nir - red,
        nir + red + 0.5
    )
    # 9. OSAVI
    vi["OSAVI"] = 1.16 * safe_divide(
        nir - red,
        nir + red + 0.16
    )
    # 10. MSR
    rvi = safe_divide(nir, red)
    vi["MSR"] = safe_divide(rvi - 1, np.sqrt(rvi + 1))
    return vi
def prepare_target(df):
    if not TARGET_IS_DATE:
        y = pd.to_numeric(df[TARGET_COL], errors="coerce")
        return y, None

    target_date = pd.to_datetime(df[TARGET_COL], errors="coerce")
    if SOWING_DATE_COL is not None and SOWING_DATE_COL in df.columns:
        sowing_date = pd.to_datetime(df[SOWING_DATE_COL], errors="coerce")
        y = (target_date - sowing_date).dt.days
        return y, "days_after_sowing"

    y = target_date.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    return y, "date_ordinal"
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def rrmse_score(y_true, y_pred):
    rmse = rmse_score(y_true, y_pred)
    mean_y = np.mean(y_true)
    if abs(mean_y) < 1e-10:
        return np.nan
    return rmse / mean_y * 100
def load_and_merge_data():
    spectrum_df = pd.read_excel(SPECTRUM_EXCEL)
    measured_df = pd.read_excel(MEASURED_EXCEL)

    required_spectrum_cols = [ID_COL, STAGE_COL]
    required_measured_cols = [ID_COL, STAGE_COL, TARGET_COL]

    if VARIETY_COL is not None:
        if VARIETY_COL in spectrum_df.columns and VARIETY_COL in measured_df.columns:
            merge_keys = [ID_COL, VARIETY_COL, STAGE_COL]
        else:
            merge_keys = [ID_COL, STAGE_COL]
            warnings.warn(
                f"没有在两个 Excel 中同时找到 {VARIETY_COL}，将只按 {ID_COL} 和 {STAGE_COL} 合并。"
            )
    else:
        merge_keys = [ID_COL, STAGE_COL]

    for col in required_spectrum_cols:
        if col not in spectrum_df.columns:
            raise ValueError(f"原始光谱 Excel 缺少列：{col}")

    for col in required_measured_cols:
        if col not in measured_df.columns:
            raise ValueError(f"实测数据 Excel 缺少列：{col}")

    # 提取波段
    band_df = get_band_reflectance(spectrum_df)

    # 计算植被指数
    vi_df = calculate_vegetation_indices(band_df)

    # 合并基础信息和 VI
    base_cols = list(dict.fromkeys(merge_keys))
    spectrum_feature_df = pd.concat(
        [spectrum_df[base_cols].reset_index(drop=True),
         vi_df.reset_index(drop=True)],
        axis=1
    )

    # 和实测数据合并
    merged_df = pd.merge(
        spectrum_feature_df,
        measured_df,
        on=merge_keys,
        how="inner"
    )

    print(f"[INFO] 原始光谱样本数：{len(spectrum_df)}")
    print(f"[INFO] 实测数据样本数：{len(measured_df)}")
    print(f"[INFO] 合并后样本数：{len(merged_df)}")

    return merged_df, list(vi_df.columns), merge_keys
def train_one_stage(stage_name, df_stage, feature_cols, merge_keys):
    y, target_mode = prepare_target(df_stage)

    data = df_stage.copy()
    data["_target_"] = y

    # 删除缺失值
    model_cols = feature_cols + ["_target_"]
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=model_cols)

    if len(data) < MIN_SAMPLES_PER_STAGE:
        print(f"[WARNING] 生育期 {stage_name} 样本数过少：{len(data)}，跳过。")
        return None, None, None

    X = data[feature_cols]
    y = data["_target_"].astype(float)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        data.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search_space = {
        "n_estimators": Integer(100, 800),
        "max_depth": Integer(3, 50),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", None]),
        "bootstrap": Categorical([True, False])
    }

    cv_splits = min(CV_SPLITS, len(X_train))
    if cv_splits < 2:
        print(f"[WARNING] 生育期 {stage_name} 训练样本过少，无法交叉验证，跳过。")
        return None, None, None

    cv = KFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    bayes_rf = BayesSearchCV(
        estimator=rf,
        search_spaces=search_space,
        n_iter=N_ITER,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0
    )

    print(f"\n[INFO] 开始训练生育期模型：{stage_name}")
    bayes_rf.fit(X_train, y_train)

    best_model = bayes_rf.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)
    rrmse = rrmse_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "growth_stage": stage_name,
        "n_samples": len(data),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "MAE": mae,
        "RMSE": rmse,
        "rRMSE_percent": rrmse,
        "R2": r2,
        "best_cv_score_neg_mse": bayes_rf.best_score_,
        "best_params": bayes_rf.best_params_
    }

    pred_df = data.loc[test_idx, merge_keys + [STAGE_COL]].copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])

    # 如果目标是日期 ordinal，把预测结果转回日期
    if TARGET_IS_DATE and target_mode == "date_ordinal":
        pred_df["true_date"] = pred_df["y_true"].round().astype(int).map(
            lambda x: pd.Timestamp.fromordinal(x)
        )
        pred_df["pred_date"] = pred_df["y_pred"].round().astype(int).map(
            lambda x: pd.Timestamp.fromordinal(x)
        )

    print(f"[RESULT] {stage_name}")
    print(f"MAE   = {mae:.4f}")
    print(f"RMSE  = {rmse:.4f}")
    print(f"rRMSE = {rrmse:.2f}%")
    print(f"R2    = {r2:.4f}")
    print(f"Best params = {bayes_rf.best_params_}")

    return best_model, metrics, pred_df
def main():
    make_dir(OUTPUT_DIR)

    merged_df, feature_cols, merge_keys = load_and_merge_data()

    all_metrics = []
    all_predictions = []

    stages = sorted(merged_df[STAGE_COL].dropna().unique())

    print("\n[INFO] 检测到的生育期：")
    for s in stages:
        print(f"  - {s}")

    for stage in stages:
        df_stage = merged_df[merged_df[STAGE_COL] == stage].copy()

        model, metrics, pred_df = train_one_stage(
            stage_name=stage,
            df_stage=df_stage,
            feature_cols=feature_cols,
            merge_keys=merge_keys
        )

        if model is None:
            continue

        safe_stage_name = str(stage).replace("/", "_").replace("\\", "_").replace(" ", "_")

        model_path = os.path.join(
            OUTPUT_DIR,
            f"bayes_rf_{safe_stage_name}.pkl"
        )

        joblib.dump(model, model_path)

        metrics["model_path"] = model_path

        all_metrics.append(metrics)
        all_predictions.append(pred_df)

    if len(all_metrics) > 0:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df["best_params"] = metrics_df["best_params"].astype(str)
        metrics_path = os.path.join(OUTPUT_DIR, "metrics_by_growth_stage.csv")
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        pred_all_df = pd.concat(all_predictions, axis=0)
        pred_path = os.path.join(OUTPUT_DIR, "predictions_by_growth_stage.csv")
        pred_all_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    else:
        print("[ERROR] ")
if __name__ == "__main__":
    main()