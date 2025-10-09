import yaml

class RiskTieringEngine:
    def __init__(self):
        self.rules = yaml.safe_load(open("config/risk_rules.yaml"))["rules"]
        self.tiers = yaml.safe_load(open("config/risk_tiering.yaml"))["risk_levels"]

    def evaluate(self, model_metadata):
        for rule in self.rules:
            conditions = rule["if"]
            match = all(
                model_metadata.get(k) in v if isinstance(v, list) else model_metadata.get(k) == v
                for k, v in conditions.items()
            )
            if match:
                return rule["then"]
        return "low"  # default

    def get_required_controls(self, risk_level):
        for tier in self.tiers:
            if tier["name"] == risk_level:
                return tier["controls"]
