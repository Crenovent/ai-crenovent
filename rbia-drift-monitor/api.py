from fastapi import FastAPI, UploadFile
import pandas as pd
from runner import run_drift_checks
from quarantine import auto_quarantine

app = FastAPI()

@app.post("/drift/check")
def drift_check(model_name: str, model_version: str, tenant_id: str, baseline: UploadFile, current: UploadFile):
    train_df = pd.read_csv(baseline.file)
    prod_df = pd.read_csv(current.file)
    report = run_drift_checks(train_df, prod_df)
    if report["status"] == "FAIL":
        auto_quarantine(model_name, model_version, tenant_id, reason="Drift threshold exceeded")
    return report
