from pydantic import BaseModel, computed_field


class ModelConfig(BaseModel):
    num_layers: int = 4
    num_heads: int = 16
    d_head: int = 32
    d_ff: int = 1344
    rope_theta: float = 10000

    @computed_field
    @property
    def d_model(self) -> int:
        return int(
            self.num_heads * self.d_head
        )


class OptConfig(BaseModel):
    max_lr: float = 1e-3
    min_lr: float = 1e-3
    warmup_iters: int = 0


class TrainingConfig(BaseModel):
    context_length: int = 256
    batch_size: int = 32
    num_tokens: int = 327_000_000
    valid_num_tokens: int = 327_000

    @computed_field
    @property
    def num_iters(self) -> int:
        return int(
            self.num_tokens / self.batch_size / self.context_length
        )

    @computed_field
    @property
    def valid_num_iters(self) -> int:
        return int(
            self.valid_num_tokens / self.batch_size / self.context_length
        )


class DatasetConfig(BaseModel):
    train_dataset_path: str
    valid_dataset_path: str
    vocab_path: str
    merges_path: str


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig = ModelConfig()
    opt: OptConfig = OptConfig()
    training: TrainingConfig = TrainingConfig()
    ckpt_dir: str
    device: str = "cpu"
    ckpt_every: int = 2000