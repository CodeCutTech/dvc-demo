import hydra
from omegaconf import DictConfig
from process_data import process_data
from segment import segment


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    process_data(config)
    segment(config)


if __name__ == "__main__":
    main()
