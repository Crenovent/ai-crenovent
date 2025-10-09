from orchestrator_api import disable_model

def auto_quarantine(model_name, version, tenant_id, reason):
    disable_model(model_name, tenant_id)
    log_event({
        "event": "model_quarantined",
        "model_name": model_name,
        "version": version,
        "reason": reason
    })
