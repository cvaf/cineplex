from .constants import SENTENCE_EMBEDDING_SHAPE, KEPT_GENRES


class Config:
    def __init__(
        self,
        num_epochs: int = 100,
        learning_rate: float = 0.1,
        gamma: float = 0.7,
        decision_threshold: float = 0.1,
        layer_sizes: list = [256, 128, 64],
        batch_size: int = 64,
        seed: int = 42,
        preload: bool = True,
    ):

        self.seed = seed
        self.preload = preload
        self.batch_size = batch_size

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.decision_threshold = decision_threshold
        self.layer_sizes = layer_sizes
        self.input_size = SENTENCE_EMBEDDING_SHAPE[0] * SENTENCE_EMBEDDING_SHAPE[1] * 2
        self.output_size = len(KEPT_GENRES)

    def model_params(self):
        return (
            self.input_size,
            self.layer_sizes,
            self.output_size,
            self.num_epochs,
            self.learning_rate,
            self.gamma,
            self.decision_threshold,
            self.seed,
            self.preload,
        )
