# main.py
import yaml
from model.trainer import train

if __name__ == '__main__':
    with open("configs/train_configs.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)
