from paddlemetrics.text.bleu import BLEUScore
from paddlemetrics.text.cer import CharErrorRate
from paddlemetrics.text.chrf import CHRFScore
from paddlemetrics.text.edit import EditDistance
from paddlemetrics.text.eed import ExtendedEditDistance
from paddlemetrics.text.mer import MatchErrorRate
from paddlemetrics.text.perplexity import Perplexity
from paddlemetrics.text.rouge import ROUGEScore
from paddlemetrics.text.sacre_bleu import SacreBLEUScore
from paddlemetrics.text.squad import SQuAD
from paddlemetrics.text.ter import TranslationEditRate
from paddlemetrics.text.wer import WordErrorRate
from paddlemetrics.text.wil import WordInfoLost
from paddlemetrics.text.wip import WordInfoPreserved
from paddlemetrics.utils.imports import _TRANSFORMERS_GREATER_EQUAL_4_4

__all__ = [
    "BLEUScore",
    "CHRFScore",
    "CharErrorRate",
    "EditDistance",
    "ExtendedEditDistance",
    "MatchErrorRate",
    "Perplexity",
    "ROUGEScore",
    "SQuAD",
    "SacreBLEUScore",
    "TranslationEditRate",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]
if _TRANSFORMERS_GREATER_EQUAL_4_4:
    from paddlemetrics.text.bert import BERTScore
    from paddlemetrics.text.infolm import InfoLM

    __all__ += ["BERTScore", "InfoLM"]
