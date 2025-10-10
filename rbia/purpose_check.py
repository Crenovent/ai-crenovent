import psycopg2, yaml

class PurposeBindingEngine:
    def __init__(self):
        self.config = yaml.safe_load(open("config/purpose_binding.yaml"))

    def check(self, workflow_purpose, dataset_id):
        conn = psycopg2.connect("dbname=rbia")
        cur = conn.cursor()
        cur.execute("SELECT data_purpose FROM rbia_data_provenance WHERE dataset_id=%s", (dataset_id,))
        data_purpose = cur.fetchone()[0]

        allowed = self.config["workflow_purposes"].get(workflow_purpose, {}).get("allowed_data_purposes", [])
        if data_purpose in allowed:
            return {"status": "PASS", "data_purpose": data_purpose}
        else:
            return {"status": "FAIL", "data_purpose": data_purpose}
