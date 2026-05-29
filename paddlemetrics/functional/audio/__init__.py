from paddlemetrics.functional.audio.pit import (permutation_invariant_training,
                                               pit_permutate)
from paddlemetrics.functional.audio.sdr import (
    scale_invariant_signal_distortion_ratio, signal_distortion_ratio,
    source_aggregated_signal_distortion_ratio)
from paddlemetrics.functional.audio.snr import (
    complex_scale_invariant_signal_noise_ratio,
    scale_invariant_signal_noise_ratio, signal_noise_ratio)
from paddlemetrics.utils.imports import (_GAMMATONE_AVAILABLE,
                                            _LIBROSA_AVAILABLE,
                                            _ONNXRUNTIME_AVAILABLE,
                                            _PESQ_AVAILABLE, _PYSTOI_AVAILABLE,
                                            _REQUESTS_AVAILABLE,
                                            _SCIPY_AVAILABLE)

if _SCIPY_AVAILABLE:
    import scipy.signal

    if not hasattr(scipy.signal, "hamming"):
        scipy.signal.hamming = scipy.signal.windows.hamming
__all__ = [
    "complex_scale_invariant_signal_noise_ratio",
    "permutation_invariant_training",
    "pit_permutate",
    "scale_invariant_signal_distortion_ratio",
    "scale_invariant_signal_noise_ratio",
    "signal_distortion_ratio",
    "signal_noise_ratio",
    "source_aggregated_signal_distortion_ratio",
]
if _PESQ_AVAILABLE:
    from paddlemetrics.functional.audio.pesq import \
        perceptual_evaluation_speech_quality

    __all__ += ["perceptual_evaluation_speech_quality"]
if _PYSTOI_AVAILABLE:
    from paddlemetrics.functional.audio.stoi import \
        short_time_objective_intelligibility

    __all__ += ["short_time_objective_intelligibility"]
if _GAMMATONE_AVAILABLE:
    from paddlemetrics.functional.audio.srmr import \
        speech_reverberation_modulation_energy_ratio

    __all__ += ["speech_reverberation_modulation_energy_ratio"]
if _LIBROSA_AVAILABLE and _ONNXRUNTIME_AVAILABLE:
    from paddlemetrics.functional.audio.dnsmos import \
        deep_noise_suppression_mean_opinion_score

    __all__ += ["deep_noise_suppression_mean_opinion_score"]
if _LIBROSA_AVAILABLE and _REQUESTS_AVAILABLE:
    from paddlemetrics.functional.audio.nisqa import \
        non_intrusive_speech_quality_assessment

    __all__ += ["non_intrusive_speech_quality_assessment"]