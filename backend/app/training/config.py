from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Experiment Tracking
    experiment_name: str = "agri_default_run"
    seed: int = 42

    # Data params
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Model/Optimization params
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    early_stopping_patience: int = 10

    # Paths
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "best_model.pth"
    export_dir: str = "exports"
    tensorboard_dir: str = "runs"

    # Switches
    use_amp: bool = True  # Automatic Mixed Precision
    allow_pseudo_severity: bool = False
