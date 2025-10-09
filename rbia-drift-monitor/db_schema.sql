CREATE TABLE rbia_drift_metrics (
   id SERIAL PRIMARY KEY,
   model_name TEXT,
   model_version TEXT,
   tenant_id TEXT,
   feature_name TEXT,
   psi FLOAT,
   ks FLOAT,
   drift_status TEXT,
   timestamp TIMESTAMP DEFAULT now()
);
