def before_data_access(workflow_purpose, dataset_id, tenant_id):
    result = purpose_engine.check(workflow_purpose, dataset_id)
    if result["status"] == "FAIL":
        log_purpose_violation(workflow_purpose, dataset_id, tenant_id, result)
        if POLICY["defaults"]["deny_on_mismatch"]:
            raise PermissionError(f"Purpose drift detected: {dataset_id} ({result['data_purpose']})")
    return True


def enforce_residency(tenant_id, requested_region):
    policy = residency_policy[tenant_id]
    if requested_region != policy["region"]:
        if not policy.get("allow_cross_border"):
            log_cross_border_violation(tenant_id, requested_region)
            raise PermissionError(f"Cross-border access denied for {tenant_id}")
        elif requested_region not in policy.get("allowed_regions", []):
            raise PermissionError("Requested region not in allowlist")
