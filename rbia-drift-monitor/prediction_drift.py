def prediction_psi(baseline_preds, live_preds):
    # Use PSI again for prediction drift
    return population_stability_index(baseline_preds, live_preds)
