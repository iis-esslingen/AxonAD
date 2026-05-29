import logging

import lightning as L
import numpy as np
import torch
import torchinfo
from TSB_AD.utils.dataset import ReconstructDataset
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.nn import MSELoss
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, Subset
from typing import Any
from xlstm import xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig, sLSTMBlockConfig, sLSTMLayerConfig, \
    FeedForwardConfig, xLSTMBlockStack

logger = logging.getLogger(__name__)

def create_config(window_size, embedding_dim=55):
    return xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=8, qkv_proj_blocksize=5, num_heads=4, round_proj_up_dim_up=False,
                round_proj_up_to_multiple_of=5, embedding_dim=embedding_dim,
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cuda",
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu", embedding_dim=embedding_dim),
        ),
        context_length=window_size,
        num_blocks=3,
        embedding_dim=embedding_dim,
        slstm_at=[1],
    )


class xLSTMADModule(L.LightningModule):
    """
    Anomaly detection model based on xLSTM architecture.
    The model consists of an encoder and a decoder, both implemented as stacks of xLSTM blocks.
    The input data is projected to the embedding dimension before being passed through the encoder,
    and the output from the decoder is projected back to the original feature space.
    The model is trained using mean squared error loss.

    xLSTMAD was published at the ICDM 2025 (IEEE International Conference on Data Mining).
    When using, please cite the xLSTMAD paper (preprint available here: https://arxiv.org/abs/2506.22837 )
    ```
        @INPROCEEDINGS{xlstmad,
          author={Faber, Kamil and Pietron, Marcin and Zurek, Dominik and Corizzo, Roberto},
          booktitle={2025 IEEE International Conference on Data Mining (ICDM)},
          title={xLSTMAD: A Powerful xLSTM-based Method for Anomaly Detection},
          year={2025},
          volume={},
          number={},
          pages={247-256},
          doi={10.1109/ICDM65498.2025.00032}}
    }
    ```

    Parameters:
    - embedding_dim: The dimension of the embedding space used in the xLSTM blocks.
    - features_no: The number of features in the input data.
    - window_size: The size of the input window for the model.
    - lr: The learning rate for the optimizer.
    """
    def __init__(self, embedding_dim: int, features_no: int, window_size: int, lr: float = 0.001):
        super(xLSTMADModule, self).__init__()
        self.window_size = window_size
        self.features_no = features_no
        self.lr = lr

        xlstm_cfg = create_config(window_size=window_size, embedding_dim=embedding_dim)
        self.encoder = xLSTMBlockStack(xlstm_cfg)
        self.decoder = xLSTMBlockStack(xlstm_cfg)

        self.input_projection = nn.Linear(features_no, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, features_no)
        self.gelu = nn.GELU()

        self.loss = MSELoss()
        self.val_loss = MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        projected_input = self.input_projection(x)
        encoder_output = self.encoder(projected_input)
        decoder_output = self.decoder(encoder_output)
        outputs = self.output_projection(self.gelu(decoder_output))
        return outputs

    def training_step(self, batch, batch_idx):
        x, _ = batch
        output = self.forward(x)
        loss = self.loss(output, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        output = self.forward(x)
        loss = self.val_loss(output, x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx) -> Any:
        x, target = batch
        reconstruction = self.forward(x)
        anomaly_scores = torch.mean(mse_loss(reconstruction, target, reduction='none'), dim=(1, 2))
        return anomaly_scores

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class xLSTMAD:
    def __init__(self, model: L.LightningModule, window_size: int = 100, validation_size: float = 0.2,
                 batch_size: int = 128):
        self.window_size = window_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.model = model

    def fit(self, data):
        full_train_dataset = ReconstructDataset(data, window_size=self.window_size)
        split_index = int((1 - self.validation_size) * len(full_train_dataset))

        train_data = Subset(full_train_dataset, range(split_index))
        valid_data = Subset(full_train_dataset, range(split_index, len(full_train_dataset)))

        logger.debug(f'train data size: { len(train_data)}')
        logger.debug(f'valid data size: {len(valid_data)}')

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        valid_loader = DataLoader(
            valid_data,
            batch_size=4 * self.batch_size,
            shuffle=False,
            num_workers=4
        )

        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            save_last=True,
            mode="min")

        trainer = L.Trainer(
            max_epochs=50,
            accelerator="gpu",
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=3, mode="min", min_delta=1e-3),
                checkpoint_cb],
            logger=True,
            enable_progress_bar=True,
        )

        logger.debug(f'Trainer log file {trainer.log_dir}')
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        logger.debug(f'Loading best model from {checkpoint_cb.best_model_path}')
        self.model = self.model.__class__.load_from_checkpoint(checkpoint_cb.best_model_path)

    def decision_function(self, data):
        data_loader = DataLoader(
            ReconstructDataset(data, window_size=self.window_size),
            batch_size=4 * self.batch_size,
            shuffle=False,
            num_workers=4,
        )

        trainer = L.Trainer(
            accelerator="gpu",
            logger=True,
            enable_checkpointing=False,
        )

        self.model.eval()
        with torch.no_grad():
            preds = trainer.predict(self.model, dataloaders=data_loader)

        scores = torch.concat(preds)
        if scores.shape[0] < len(data):
            logging.info("Adjusting anomaly scores length to match data length.")
            padded_decision_scores = np.zeros(len(data))
            padded_decision_scores[: self.window_size - 1] = scores[0]
            padded_decision_scores[self.window_size- 1:] = scores
            return padded_decision_scores

        return scores.numpy()

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window_size), verbose=0)
        with open(save_file, 'w') as f:
            f.write(str(model_stats))