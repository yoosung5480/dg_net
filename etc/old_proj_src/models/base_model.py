import torch
import torch.nn as nn
from pathlib import Path


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def save(self, save_dir: str, name: str = "model"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # weight
        torch.save(self.state_dict(), save_dir / f"{name}.pt")

        # config
        if hasattr(self.cfg, "save"):
            self.cfg.save(save_dir)

    @classmethod
    def load(cls, load_dir: str, name: str = "model"):
        load_dir = Path(load_dir)

        # config
        cfg = cls.ConfigClass.load(load_dir)

        # model build
        model = cls(cfg)

        # weight load
        state_dict = torch.load(load_dir / f"{name}.pt", map_location="cpu")
        model.load_state_dict(state_dict)

        return model