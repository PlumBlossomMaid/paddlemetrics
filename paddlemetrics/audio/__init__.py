from paddlemetrics.audio.pit import PermutationInvariantTraining
from paddlemetrics.audio.sdr import (ScaleInvariantSignalDistortionRatio,
                                    SignalDistortionRatio,
                                    SourceAggregatedSignalDistortionRatio)
from paddlemetrics.audio.snr import (ComplexScaleInvariantSignalNoiseRatio,
                                    ScaleInvariantSignalNoiseRatio,
                                    SignalNoiseRatio)
from paddlemetrics.utils.imports import (_GAMMATONE_AVAILABLE,
                                            _LIBROSA_AVAILABLE,
                                            _ONNXRUNTIME_AVAILABLE,
                                            _PESQ_AVAILABLE, _PYSTOI_AVAILABLE,
                                            _REQUESTS_AVAILABLE,
                                            _SCIPY_AVAILABLE,
                                            _TORCHAUDIO_AVAILABLE)

if _SCIPY_AVAILABLE:
    import scipy.signal

    if not hasattr(scipy.signal, "hamming"):
        scipy.signal.hamming = scipy.signal.windows.hamming
__all__ = [
    "ComplexScaleInvariantSignalNoiseRatio",
    "PermutationInvariantTraining",
    "ScaleInvariantSignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SignalDistortionRatio",
    "SignalNoiseRatio",
    "SourceAggregatedSignalDistortionRatio",
]
if _PESQ_AVAILABLE:
    from paddlemetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    __all__ += ["PerceptualEvaluationSpeechQuality"]
if _PYSTOI_AVAILABLE:
    from paddlemetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    __all__ += ["ShortTimeObjectiveIntelligibility"]
if _GAMMATONE_AVAILABLE and _TORCHAUDIO_AVAILABLE:
    from paddlemetrics.audio.srmr import \
        SpeechReverberationModulationEnergyRatio

    __all__ += ["SpeechReverberationModulationEnergyRatio"]
if _LIBROSA_AVAILABLE and _ONNXRUNTIME_AVAILABLE:
    from paddlemetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

    __all__ += ["DeepNoiseSuppressionMeanOpinionScore"]
if _LIBROSA_AVAILABLE and _REQUESTS_AVAILABLE:
    from paddlemetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

    __all__ += ["NonIntrusiveSpeechQualityAssessment"]
