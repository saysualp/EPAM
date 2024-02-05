import hydra
from omegaconf import DictConfig
from src.data.make_dataset import make_dataset
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import predict_model


@hydra.main(config_path="../config/config.yaml")
def run(cfg: DictConfig):

    # Step 1: Data preparation
    make_dataset(cfg)

    # Step 2: Feature building
    model_version = build_features(
        cfg, forecast_horizon=cfg.train.forecast_horizon
    )

    # Step 3: Model training
    train_model(
        cfg,
        model_version=model_version,
        forecast_horizon=cfg.train.forecast_horizon,
    )

    # Step 4: Make predictions
    predict_model(
        cfg,
        model_version=model_version,
        forecast_horizon=cfg.train.forecast_horizon,
    )


if __name__ == "__main__":
    run()
