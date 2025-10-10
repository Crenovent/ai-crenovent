import yaml
from data_drift import population_stability_index, ks_drift
from prediction_drift import prediction_psi

def run_drift_checks(train_df, prod_df, config_path="config/thresholds.yaml"):
    thresholds = yaml.safe_load(open(config_path))
    report = {"data_drift": {}, "prediction_drift": {}, "status": "PASS"}

    # Feature-level drift
    for col in train_df.columns:
        psi = population_stability_index(train_df[col], prod_df[col])
        ks = ks_drift(train_df[col], prod_df[col])
        report["data_drift"][col] = {"psi": psi, "ks": ks}
        if psi > thresholds["data_drift"]["psi_threshold"] or ks > thresholds["data_drift"]["ks_threshold"]:
            report["status"] = "FAIL"

    # Prediction drift
    if "prediction" in prod_df:
        psi_pred = prediction_psi(train_df["prediction"], prod_df["prediction"])
        report["prediction_drift"]["psi"] = psi_pred
        if psi_pred > thresholds["prediction_drift"]["psi_threshold"]:
            report["status"] = "FAIL"

    return report
