from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BaseConfig:
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "model_config.json", "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        with open(path / "model_config.json", "r") as f:
            cfg_dict = json.load(f)
        return cls(**cfg_dict)