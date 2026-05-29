from paddlemetrics.image.d_lambda import SpectralDistortionIndex
from paddlemetrics.image.d_s import SpatialDistortionIndex
from paddlemetrics.image.dists import DeepImageStructureAndTextureSimilarity
from paddlemetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from paddlemetrics.image.mifid import \
    MemorizationInformedFrechetInceptionDistance
from paddlemetrics.image.psnr import PeakSignalNoiseRatio
from paddlemetrics.image.psnrb import PeakSignalNoiseRatioWithBlockedEffect
from paddlemetrics.image.qnr import QualityWithNoReference
from paddlemetrics.image.rase import RelativeAverageSpectralError
from paddlemetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from paddlemetrics.image.sam import SpectralAngleMapper
from paddlemetrics.image.scc import SpatialCorrelationCoefficient
from paddlemetrics.image.ssim import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure)
from paddlemetrics.image.tv import TotalVariation
from paddlemetrics.image.uqi import UniversalImageQualityIndex
from paddlemetrics.image.vif import VisualInformationFidelity
from paddlemetrics.utils.imports import (_TORCH_FIDELITY_AVAILABLE,
                                            _TORCHVISION_AVAILABLE)

__all__ = [
    "DeepImageStructureAndTextureSimilarity",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "MemorizationInformedFrechetInceptionDistance",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "QualityWithNoReference",
    "RelativeAverageSpectralError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpatialCorrelationCoefficient",
    "SpatialDistortionIndex",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StructuralSimilarityIndexMeasure",
    "TotalVariation",
    "UniversalImageQualityIndex",
    "VisualInformationFidelity",
]
if _TORCH_FIDELITY_AVAILABLE:
    from paddlemetrics.image.fid import FrechetInceptionDistance
    from paddlemetrics.image.inception import InceptionScore
    from paddlemetrics.image.kid import KernelInceptionDistance

    __all__ += ["FrechetInceptionDistance", "InceptionScore", "KernelInceptionDistance"]
if _TORCHVISION_AVAILABLE:
    from paddlemetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from paddlemetrics.image.perceptual_path_length import PerceptualPathLength

    __all__ += ["LearnedPerceptualImagePatchSimilarity", "PerceptualPathLength"]
