import os
from .constants import WORD_EMBEDDING_SHAPE, KEPT_GENRES, RESULTS_FOLDER
from datetime import datetime


class Config:
    def __init__(
        self,
        num_epochs: int = 100,
        learning_rate: float = 0.1,
        gamma: float = 0.7,
        decay_step_size: int = 1,
        decision_threshold: float = 0.2,
        layer_sizes: list = [512, 256, 128],
        batch_size: int = 64,
        seed: int = 42,
        preload: bool = True,
    ):

        self.seed = seed
        self.preload = preload
        self.batch_size = batch_size
        self.results_path = os.path.join(
            RESULTS_FOLDER,
            datetime.now().strftime("%Y-%m-%d--%H-%M-%S"),
        )

        # Modeling parameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decay_step_size = decay_step_size
        self.decision_threshold = decision_threshold
        self.layer_sizes = layer_sizes
        self.input_size = WORD_EMBEDDING_SHAPE[0] * 2
        self.output_size = len(KEPT_GENRES)

    def model_params(self):
        return (
            self.input_size,
            self.layer_sizes,
            self.output_size,
            self.num_epochs,
            self.learning_rate,
            self.gamma,
            self.decay_step_size,
            self.decision_threshold,
            self.results_path,
            self.seed,
            self.preload,
        )
