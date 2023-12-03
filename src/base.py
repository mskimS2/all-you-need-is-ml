import json
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs/yaml", config_name="catboost")
def model_config(cfg: DictConfig) -> DictConfig:
    print(cfg)
    return cfg
    
    
if __name__ == "__main__":
    model_config()