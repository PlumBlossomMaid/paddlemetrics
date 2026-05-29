import copy
import math
import os
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import paddle

from paddlemetrics.utils import rank_zero_info
from paddlemetrics.utils.imports import (_LIBROSA_AVAILABLE,
                                            _REQUESTS_AVAILABLE)
from paddlemetrics.utils.compat import pack_padded_sequence, pad_packed_sequence

if _LIBROSA_AVAILABLE and _REQUESTS_AVAILABLE:
    import librosa
    import requests
else:
    librosa, requests = None, None
__doctest_requires__ = {
    ("non_intrusive_speech_quality_assessment",): ["librosa", "requests"]
}
NISQA_DIR = "~/.paddlemetrics/NISQA"


def non_intrusive_speech_quality_assessment(
    preds: paddle.Tensor, fs: int
) -> paddle.Tensor:
    """`Non-Intrusive Speech Quality Assessment`_ (NISQA v2.0) [1], [2].

    .. hint::
        Usingsing this metric requires you to have ``librosa`` and ``requests`` installed. Install as
        ``pip install librosa requests``.

    Args:
        preds: float tensor with shape ``(...,time)``
        fs: sampling frequency of input

    Returns:
        Float tensor with shape ``(...,5)`` corresponding to overall MOS, noisiness, discontinuity, coloration and
        loudness in that order

    Raises:
        ModuleNotFoundError:
            If ``librosa`` or ``requests`` are not installed
        RuntimeError:
            If the input is too short, causing the number of mel spectrogram windows to be zero
        RuntimeError:
            If the input is too long, causing the number of mel spectrogram windows to exceed the maximum allowed

    Example:
        >>> import paddle
        >>> from paddlemetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
        >>> _ = paddle.seed(42)
        >>> preds = paddle.randn([16000])
        >>> non_intrusive_speech_quality_assessment(preds, 16000)
        tensor([1.0433, 1.9545, 2.6087, 1.3460, 1.7117])

    References:
        - [1] G. Mittag and S. Möller, "Non-intrusive speech quality assessment for super-wideband speech communication
          networks", in Proc. ICASSP, 2019.
        - [2] G. Mittag, B. Naderi, A. Chehadi and S. Möller, "NISQA: A deep CNN-self-attention model for
          multidimensional speech quality prediction with crowdsourced datasets", in Proc. INTERSPEECH, 2021.

    """
    if not _LIBROSA_AVAILABLE or not _REQUESTS_AVAILABLE:
        raise ModuleNotFoundError(
            "NISQA metric requires that librosa and requests are installed. Install as `pip install librosa requests`."
        )
    model, args = _load_nisqa_model()
    if not isinstance(fs, int) or fs <= 0:
        raise ValueError(
            f"Argument `fs` expected to be a positive integer, but got {fs}"
        )
    model.eval()
    x = preds.reshape([-1, preds.shape[-1]])
    x = _get_librosa_melspec(x.numpy(), fs, args)
    x, n_wins = _segment_specs(paddle.to_tensor(x), args)
    with paddle.no_grad():
        x = model(x, n_wins.expand([x.shape[0]]))
    return x.reshape([*preds.shape[:-1], 5])


@lru_cache
def _load_nisqa_model() -> tuple[paddle.nn.Layer, dict[str, Any]]:
    """Load NISQA model and its parameters.

    Returns:
        Tuple ``(model,args)`` where ``model`` is the NISQA model and ``args`` is a dictionary with all its parameters

    """
    model_path = os.path.expanduser(os.path.join(NISQA_DIR, "nisqa.tar"))
    if not os.path.exists(model_path):
        _download_weights()
    checkpoint = paddle.load(path=str(model_path))
    args = checkpoint["args"]
    model = _NISQADIM(args)
    model.set_state_dict(checkpoint["model_state_dict"])
    return model, args


def _download_weights() -> None:
    """Download NISQA model weights."""
    url = (
        "https://github.com/gabrielmittag/NISQA/raw/refs/heads/master/weights/nisqa.tar"
    )
    nisqa_dir = os.path.expanduser(NISQA_DIR)
    os.makedirs(nisqa_dir, exist_ok=True)
    saveto = os.path.join(nisqa_dir, "nisqa.tar")
    if os.path.exists(saveto):
        return
    rank_zero_info(f"downloading {url} to {saveto}")
    myfile = requests.get(url)
    with open(saveto, "wb") as f:
        f.write(myfile.content)


class _NISQADIM(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.cnn = _Framewise(args)
        self.time_dependency = _TimeDependency(args)
        pool = _Pooling(args)
        self.pool_layers = _get_clones(pool, 5)

    def forward(self, x: paddle.Tensor, n_wins: paddle.Tensor) -> paddle.Tensor:
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        return paddle.concat(out, axis=1)


class _Framewise(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.model = _AdaptCNN(args)

    def forward(self, x: paddle.Tensor, n_wins: paddle.Tensor) -> paddle.Tensor:
        x_packed = pack_padded_sequence(
            x, n_wins, batch_first=True, enforce_sorted=False
        )
        packed_data, batch_sizes, sorted_indices, unsorted_indices = x_packed
        x = self.model(packed_data.unsqueeze(1))
        x = (x, batch_sizes, sorted_indices, unsorted_indices)
        x, _ = pad_packed_sequence(
            x, batch_first=True, padding_value=0.0, total_length=int(n_wins.max())
        )
        return x


class _AdaptCNN(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.pool_1 = args["cnn_pool_1"]
        self.pool_2 = args["cnn_pool_2"]
        self.pool_3 = args["cnn_pool_3"]
        self.dropout = paddle.nn.Dropout2D(p=args["cnn_dropout"])
        cnn_pad = (1, 0) if args["cnn_kernel_size"][0] == 1 else (1, 1)
        self.conv1 = paddle.nn.Conv2D(
            1, args["cnn_c_out_1"], args["cnn_kernel_size"], padding=cnn_pad
        )
        self.bn1 = paddle.nn.BatchNorm2D(num_features=self.conv1._out_channels)
        self.conv2 = paddle.nn.Conv2D(
            self.conv1._out_channels,
            args["cnn_c_out_2"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn2 = paddle.nn.BatchNorm2D(num_features=self.conv2._out_channels)
        self.conv3 = paddle.nn.Conv2D(
            self.conv2._out_channels,
            args["cnn_c_out_3"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn3 = paddle.nn.BatchNorm2D(num_features=self.conv3._out_channels)
        self.conv4 = paddle.nn.Conv2D(
            self.conv3._out_channels,
            args["cnn_c_out_3"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn4 = paddle.nn.BatchNorm2D(num_features=self.conv4._out_channels)
        self.conv5 = paddle.nn.Conv2D(
            self.conv4._out_channels,
            args["cnn_c_out_3"],
            args["cnn_kernel_size"],
            padding=cnn_pad,
        )
        self.bn5 = paddle.nn.BatchNorm2D(num_features=self.conv5._out_channels)
        self.conv6 = paddle.nn.Conv2D(
            self.conv5._out_channels,
            args["cnn_c_out_3"],
            (args["cnn_kernel_size"][0], args["cnn_pool_3"][1]),
            padding=(1, 0),
        )
        self.bn6 = paddle.nn.BatchNorm2D(num_features=self.conv6._out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = paddle.nn.functional.relu(self.bn1(self.conv1(x)))
        x = paddle.nn.functional.adaptive_max_pool2d(x, output_size=self.pool_1)
        x = paddle.nn.functional.relu(self.bn2(self.conv2(x)))
        x = paddle.nn.functional.adaptive_max_pool2d(x, output_size=self.pool_2)
        x = self.dropout(x)
        x = paddle.nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = paddle.nn.functional.relu(self.bn4(self.conv4(x)))
        x = paddle.nn.functional.adaptive_max_pool2d(x, output_size=self.pool_3)
        x = self.dropout(x)
        x = paddle.nn.functional.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = paddle.nn.functional.relu(self.bn6(self.conv6(x)))
        return x.reshape([-1, self.conv6._out_channels * self.pool_3[0]])


class _TimeDependency(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.model = _SelfAttention(args)

    def forward(self, x: paddle.Tensor, n_wins: paddle.Tensor) -> paddle.Tensor:
        return self.model(x, n_wins)


class _SelfAttention(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        encoder_layer = _SelfAttentionLayer(args)
        self.norm1 = paddle.nn.LayerNorm(args["td_sa_d_model"])
        self.linear = paddle.nn.Linear(
            args["cnn_c_out_3"] * args["cnn_pool_3"][0], args["td_sa_d_model"]
        )
        self.layers = _get_clones(encoder_layer, args["td_sa_num_layers"])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                paddle.nn.initializer.XavierUniform(p)

    def forward(
        self, src: paddle.Tensor, n_wins: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        src = self.linear(src)
        output = src.transpose([1, 0, 2])
        output = self.norm1(output)
        for mod in self.layers:
            output, n_wins = mod(output, n_wins)
        return output.transpose([1, 0, 2]), n_wins


class _SelfAttentionLayer(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.self_attn = paddle.nn.MultiHeadAttention(
            args["td_sa_d_model"], args["td_sa_nhead"], args["td_sa_dropout"]
        )
        self.linear1 = paddle.nn.Linear(args["td_sa_d_model"], args["td_sa_h"])
        self.dropout = paddle.nn.Dropout(args["td_sa_dropout"])
        self.linear2 = paddle.nn.Linear(args["td_sa_h"], args["td_sa_d_model"])
        self.norm1 = paddle.nn.LayerNorm(args["td_sa_d_model"])
        self.norm2 = paddle.nn.LayerNorm(args["td_sa_d_model"])
        self.dropout1 = paddle.nn.Dropout(args["td_sa_dropout"])
        self.dropout2 = paddle.nn.Dropout(args["td_sa_dropout"])
        self.activation = paddle.nn.functional.relu

    def forward(
        self, src: paddle.Tensor, n_wins: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        mask = paddle.arange(src.shape[0])[None, :] < n_wins[:, None]
        src2 = self.self_attn(src, src, src, key_padding_mask=~mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, n_wins


class _Pooling(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.model = _PoolAttFF(args)

    def forward(self, x: paddle.Tensor, n_wins: paddle.Tensor) -> paddle.Tensor:
        return self.model(x, n_wins)


class _PoolAttFF(paddle.nn.Layer):
    def __init__(self, args: dict[str, Any]) -> None:
        super().__init__()
        self.linear1 = paddle.nn.Linear(
            args["td_sa_d_model"], args["pool_att_h"]
        )
        self.linear2 = paddle.nn.Linear(args["pool_att_h"], 1)
        self.linear3 = paddle.nn.Linear(args["td_sa_d_model"], 1)
        self.activation = paddle.nn.functional.relu
        self.dropout = paddle.nn.Dropout(args["pool_att_dropout"])

    def forward(self, x: paddle.Tensor, n_wins: paddle.Tensor) -> paddle.Tensor:
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose([2, 1, 0])
        mask = paddle.arange(att.shape[2])[None, :] < n_wins[:, None]
        att[~mask.unsqueeze(1)] = float("-inf")
        att = paddle.nn.functional.softmax(att, axis=2)
        x = paddle.bmm(att, x.transpose([1, 0, 2]))
        x = x.squeeze(1)
        return self.linear3(x)


def _get_librosa_melspec(y: np.ndarray, sr: int, args: dict[str, Any]) -> np.ndarray:
    """Compute mel spectrogram from waveform using librosa.

    Args:
        y: waveform with shape ``(batch_size,time)``
        sr: sampling rate
        args: dictionary with all NISQA parameters

    Returns:
        Mel spectrogram with shape ``(batch_size,n_mels,n_frames)``

    """
    hop_length = int(sr * args["ms_hop_length"])
    win_length = int(sr * args["ms_win_length"])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Empty filters detected in mel frequency basis"
        )
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            S=None,
            n_fft=args["ms_n_fft"],
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=1.0,
            n_mels=args["ms_n_mels"],
            fmin=0.0,
            fmax=args["ms_fmax"],
            htk=False,
            norm="slaney",
        )
    return np.stack(
        [librosa.amplitude_to_db(m, ref=1.0, amin=0.0001, top_db=80.0) for m in melspec]
    )


def _segment_specs(
    x: paddle.Tensor, args: dict[str, Any]
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Segment mel spectrogram into overlapping windows.

    Args:
        x: mel spectrogram with shape ``(batch_size,n_mels,n_frames)``
        args: dictionary with all NISQA parameters

    Returns:
        Tuple ``(x_padded,n_wins)```, where ``x_padded`` is the segmented mel spectrogram with shape
        ``(batch_size,max_length,n_mels,seg_length)`` where the second dimension is the number of windows and was
        padded to ``max_length``, and ``n_wins`` is the number of windows and is 0-dimensional

    """
    seg_length = args["ms_seg_length"]
    seg_hop = args["ms_seg_hop_length"]
    max_length = args["ms_max_segments"]
    n_wins = x.shape[2] - (seg_length - 1)
    if n_wins < 1:
        raise RuntimeError("Input signal is too short.")
    idx1 = paddle.arange(seg_length)
    idx2 = paddle.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose([2, 1, 0])[:, idx3, :].transpose([3, 2, 0, 1])
    x = x[:, ::seg_hop]
    n_wins = math.ceil(n_wins / seg_hop)
    if max_length < n_wins:
        raise RuntimeError(
            "Maximum number of mel spectrogram windows exceeded. Use shorter audio."
        )
    x_padded = paddle.zeros([x.shape[0], max_length, x.shape[2], x.shape[3]])
    x_padded[:, :n_wins] = x
    return x_padded, paddle.to_tensor(n_wins)


def _get_clones(module: paddle.nn.Layer, n: int) -> paddle.nn.LayerList:
    """Create ``n`` copies of a module."""
    return paddle.nn.LayerList([copy.deepcopy(module) for i in range(n)])