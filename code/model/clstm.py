import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import WEIGHTS_DIR

_NUM_MEL_BANDS = 80
_NUM_FFT_FRAME_LENGTHS = 3


class SpectrogramNormalizer(nn.Module):
    """Normalizes log-Mel spectrograms to zero mean and unit variance per bin."""

    def __init__(self, load_moments: bool = True):
        super().__init__()
        self.mean = nn.Parameter(
            torch.zeros(
                (_NUM_MEL_BANDS, _NUM_FFT_FRAME_LENGTHS),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )
        self.std = nn.Parameter(
            torch.ones(
                (_NUM_MEL_BANDS, _NUM_FFT_FRAME_LENGTHS),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )
        if load_moments:
            self.load_state_dict(
                torch.load(pathlib.Path(WEIGHTS_DIR, "spectrogram_normalizer.bin"))
            )

    def forward(self, x: torch.Tensor):
        """Normalizes log-Mel spectrograms to zero mean and unit variance per bin.

        Args:
            x: 44.1kHz waveforms as float32 [batch_size, num_frames, num_mel_bands (80), num_fft_frame_lengths (3)].
        Returns:
            Normalized input (same shape).
        """
        return (x - self.mean) / self.std


_FEATURE_CONTEXT_RADIUS_1 = 7
_FEATURE_CONTEXT_RADIUS_2 = 3


class PlacementCLSTM(nn.Module):
    """Predicts placement scores from log-Mel spectrograms."""

    def __init__(self, load_pretrained_weights: bool = False):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 10, (7, 3))
        self.maxpool0 = nn.MaxPool2d((1, 3), (1, 3))
        self.conv1 = nn.Conv2d(10, 20, (3, 3))
        self.maxpool1 = nn.MaxPool2d((1, 3), (1, 3))
        self.dense0 = nn.Linear(1400, 256)
        self.dense1 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(input_size=165,
                            hidden_size=200,
                            num_layers=2,
                            bidirectional=False,
                            batch_first=True,
                            dropout=0.5)
        
        if load_pretrained_weights:
            self.load_state_dict(
                torch.load(pathlib.Path(WEIGHTS_DIR, "placement_cnn_ckpt_56000.bin"))
            )
        condition_dim = 5
        self.conditions = nn.Sequential(
            nn.Linear(1,condition_dim*2),
            nn.ReLU(),
            nn.Linear(condition_dim*2,condition_dim),
            nn.ReLU()
        )

    def conv(self, x: torch.Tensor):
        # x is b, 3, 15, 80

        # Conv 0
        x = self.conv0(x)
        x = F.relu(x)
        x = self.maxpool0(x)

        # Conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        return x

    def dense(self, x_conv_diff: torch.Tensor):
        # x is b, 112, 1125
        bs = x_conv_diff.shape[0]
        # Dense 0
        x_conv_diff = x_conv_diff.reshape( (-1, 1400) )
        x = self.dense0(x_conv_diff)
        x = F.relu(x)
        x = self.dropout(x)

        # Dense 1
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output
        x = self.output(x)
        x = x.reshape((bs, -1) )

        return x

    def forward(
        self,
        x: torch.Tensor,
        difficulties: torch.Tensor,
        output_logits: bool = False,
        conv_chunk_size: int = 256,
        dense_chunk_size: int = 256,
    ):
        """Predicts placement scores from normalized log-Mel spectrograms.

        Args:
            x: Normalized Log-Mel spectrograms as float32 [batch_size, num_frames, num_mel_bands (80), num_fft_frame_lengths (3)].
            difficulties: DDR difficulty labels as int64 [batch_size]
            output_logits: If True, output raw logits instead of sigmoid scores (default).

        Returns:
            Placement scores (or logits) as float32 [batch_size, num_frames].
        """

        # TODO: Proper batch support for this module

        # x is B, t, 80, 3
        num_timesteps = x.shape[1]
        bs = x.shape[0]

        # print(x.shape)

        # Pad features
        x_padded = F.pad(
            x, (0, 0, 0, 0, _FEATURE_CONTEXT_RADIUS_1, _FEATURE_CONTEXT_RADIUS_1, 0, 0)
        )

        # print(x_padded.size())

        # Convolve
        x_padded = x_padded.permute(0, 3, 1, 2)
        # x_conv = []
        # for i in range(0, num_timesteps, conv_chunk_size):
        #     x_chunk = x_padded[
        #         :, :, i : i + conv_chunk_size + _FEATURE_CONTEXT_RADIUS_1 * 2
        #     ]
        #     # print(f"x_chunk : {x_chunk.shape}")
        #     x_chunk_conv = self.conv(x_chunk)
        #     # print(f"x_chunk_conv : {x_chunk_conv.shape}")
        #     assert x_chunk_conv.shape[1] > _FEATURE_CONTEXT_RADIUS_2 * 2
        #     if i == 0:
        #         x_conv.append(x_chunk_conv[:, :, :_FEATURE_CONTEXT_RADIUS_2])
        #     x_conv.append(
        #         x_chunk_conv[:, :, _FEATURE_CONTEXT_RADIUS_2:-_FEATURE_CONTEXT_RADIUS_2]
        #     )
        # x_conv.append(x_chunk_conv[:, :, -_FEATURE_CONTEXT_RADIUS_2:])
        # for "small" timestamp sizes, all of the above boils down to this.
        # x_conv = torch.cat(x_conv, dim=2)
        x_conv = self.conv(x_padded[:, :, :, :])
        # print(f"x_conv: {x_conv.shape}")
        x_conv = x_conv.permute(0, 2, 3, 1)
        # print(f"x_conv: {x_conv.shape}")
        x_conv = x_conv.reshape(bs, -1, 160) 
        # print(f"x_conv: {x_conv.shape}") # (B, 118, 160)

        # LSTM
        d = difficulties[:]
        doh = self.conditions(d.unsqueeze(1)).float().unsqueeze(1)
        doh = doh.repeat((1,x_conv.shape[1],1))
        lstm_input = torch.cat((x_conv, doh), dim=2) # (B, 118, 165)
        lstm_output = self.lstm(lstm_input)[0]
        # print(f"lstm_output, {lstm_output.shape}")
        # Dense
        logits = []
        for i in range(0, num_timesteps, dense_chunk_size):
            # TODO: Turn this into a convolutional layer?
            # NOTE: Pytorch didn't like this as of 20-03-15:
            # https://github.com/pytorch/pytorch/pull/33073
            x_chunk = []
            for j in range(i, i + dense_chunk_size):
                if j >= num_timesteps:
                    break
                x_chunk.append(lstm_output[:, j : j + 1 + _FEATURE_CONTEXT_RADIUS_2 * 2])

            x_chunk = torch.stack(x_chunk, dim=1)
            # print(f"x_chunk : {x_chunk.shape}")
            x_chunk = x_chunk.reshape(bs, -1, 1400)
            # print(f"x_chunk : {x_chunk.shape}")

            # Compute dense layer for each difficulty
            logits_diffs = []

            x_chunk_dense = self.dense(x_chunk)
            # print(f"x_chunk_dense : {x_chunk_dense.shape}")
            logits_diffs.append(x_chunk_dense)

            # print(len(logits_diffs), logits_diffs[0].shape)
            logits_diffs = torch.stack(logits_diffs, dim=0)
            logits.append(logits_diffs)
            # print(f"logits {logits[0].shape}")
        logits = torch.cat(logits, dim=1)
        # print(f"logits_after {logits[0].shape} {logits[0]}")

        if output_logits:
            return logits
        else:
            ret = torch.sigmoid(logits)
            # print(f"ret {ret.shape} {ret}")
            return ret.squeeze(0)


if __name__ == "__main__":
    cnn = PlacementCLSTM()
    B = 64
    x = torch.rand([64,112, 80, 3])
    diff = torch.rand([64])
    # print(cnn(x,difficulties=diff).shape)