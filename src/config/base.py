from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path=".", config_name="test")
def model_config(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
if __name__ == "__main__":
    model_config()