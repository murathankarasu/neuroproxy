"""EfficientPhys model definition, vendored from rPPG-Toolbox.

Source: https://github.com/ubicomplab/rPPG-Toolbox
        neural_methods/model/EfficientPhys.py
Paper:  Liu et al., "EfficientPhys: Enabling Simple, Fast and Accurate
        Camera-Based Cardiac Measurement", WACV 2023.

LICENCE -- Responsible AI Source Code License v1.1 (RAIL).
Commercial use is permitted. Behavioural restrictions are NOT, and section 3.2
of that licence requires them to be **passed through contractually to every
downstream user**. The restricted uses include:

  - surveillance, and inferring identity attributes including health and
    medical conditions
  - determining insurance premiums, or denying insurance claims
  - diagnosing a medical condition without human oversight
  - predicting criminal behaviour from personal characteristics, explicitly
    including heart rate, perspiration and breathing

These overlap almost exactly with the exclusions the design doc already sets
for the product (section 11). The obligation that is easy to miss is the
pass-through: shipping anything derived from this code obliges NeuroProxy's own
terms of service to carry the same restrictions. See docs/datasets.md.

Vendored rather than imported so the pipeline has no dependency on the toolbox
package. Kept byte-faithful to the original apart from formatting, so that the
released checkpoints load without surprises.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    """Temporal shift module: mixes adjacent frames without extra parameters."""

    def __init__(self, n_segment=10, fold_div=3):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold : 2 * fold] = x[:, :-1, fold : 2 * fold]
        out[:, :, 2 * fold :] = x[:, :, 2 * fold :]
        return out.view(nt, c, h, w)


class EfficientPhys(nn.Module):
    def __init__(
        self,
        in_channels=3,
        nb_filters1=32,
        nb_filters2=64,
        kernel_size=3,
        dropout_rate1=0.25,
        dropout_rate2=0.5,
        pool_size=(2, 2),
        nb_dense=128,
        frame_depth=20,
        img_size=72,
        channel="raw",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense
        self.frame_depth = frame_depth
        self.img_size = img_size

        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        self.motion_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size, padding=(1, 1), bias=True)
        self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size, bias=True)
        self.motion_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size, padding=(1, 1), bias=True)
        self.motion_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size, bias=True)

        self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, kernel_size=1, padding=(0, 0), bias=True)
        self.attn_mask_2 = Attention_mask()

        self.avg_pooling_1 = nn.AvgPool2d(pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(pool_size)

        self.dropout_1 = nn.Dropout(dropout_rate1)
        self.dropout_2 = nn.Dropout(dropout_rate1)
        self.dropout_3 = nn.Dropout(dropout_rate1)
        self.dropout_4 = nn.Dropout(dropout_rate2)

        dense_in = {36: 3136, 72: 16384, 96: 30976}.get(img_size)
        if dense_in is None:
            raise ValueError("unsupported img_size {}".format(img_size))
        self.final_dense_1 = nn.Linear(dense_in, nb_dense, bias=True)
        self.final_dense_2 = nn.Linear(nb_dense, 1, bias=True)
        self.batch_norm = nn.BatchNorm2d(3)
        self.channel = channel

    def forward(self, inputs, params=None):
        # The model differences frames itself, so it consumes T+1 frames and
        # emits T samples. Callers must supply the extra frame.
        inputs = torch.diff(inputs, dim=0)
        inputs = self.batch_norm(inputs)

        network_input = self.TSM_1(inputs)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        return self.final_dense_2(d11)


def load_pretrained(path, img_size=72, frame_depth=20) -> "EfficientPhys":
    """Load a released checkpoint, stripping the DataParallel `module.` prefix."""
    model = EfficientPhys(img_size=img_size, frame_depth=frame_depth)
    state = torch.load(str(path), map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model
