import json
import uuid
from pathlib import Path


class JSONLogger:
    def __init__(self, exp_name=None):
        self.exp_name = exp_name or uuid.uuid4()
        self.log_file = f"experiments/{self.exp_name}.json"
        Path("experiments").mkdir(exist_ok=True)

        self.data = {"hyperparams": {}, "metrics": []}

    def log_params(self, params):
        self.data["hyperparams"] = params
        self._save()

    def log_metrics(self, step, **metrics):
        self.data["metrics"].append(
            {
                "step": step,
                **metrics,
            }
        )
        self._save()

    def _save(self):
        with open(self.log_file, "w") as f:
            json.dump(self.data, f, indent=2)


if __name__ == "__main__":
    logger = JSONLogger("experiment_001")
    logger.log_params({"lr": 0.001, "batch_size": 32})
    logger.log_metric({"loss": 0.5, "acc": 0.95}, step=1)
