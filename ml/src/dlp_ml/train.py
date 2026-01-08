import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=== ML Config ===")
    print(OmegaConf.to_yaml(cfg))
    # TODO: load dataset_id, read manifest.csv, start training

if __name__ == "__main__":
    main()
