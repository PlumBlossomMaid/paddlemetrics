from paddlemetrics.functional.text.bleu import bleu_score
from paddlemetrics.functional.text.cer import char_error_rate
from paddlemetrics.functional.text.chrf import chrf_score
from paddlemetrics.functional.text.edit import edit_distance
from paddlemetrics.functional.text.eed import extended_edit_distance
from paddlemetrics.functional.text.mer import match_error_rate
from paddlemetrics.functional.text.perplexity import perplexity
from paddlemetrics.functional.text.rouge import rouge_score
from paddlemetrics.functional.text.sacre_bleu import sacre_bleu_score
from paddlemetrics.functional.text.squad import squad
from paddlemetrics.functional.text.ter import translation_edit_rate
from paddlemetrics.functional.text.wer import word_error_rate
from paddlemetrics.functional.text.wil import word_information_lost
from paddlemetrics.functional.text.wip import word_information_preserved
from paddlemetrics.utils.imports import _TRANSFORMERS_GREATER_EQUAL_4_4

__all__ = [
    "bleu_score",
    "char_error_rate",
    "chrf_score",
    "edit_distance",
    "extended_edit_distance",
    "match_error_rate",
    "perplexity",
    "rouge_score",
    "sacre_bleu_score",
    "squad",
    "translation_edit_rate",
    "word_error_rate",
    "word_information_lost",
    "word_information_preserved",
]
if _TRANSFORMERS_GREATER_EQUAL_4_4:
    from paddlemetrics.functional.text.bert import bert_score
    from paddlemetrics.functional.text.infolm import infolm

    __all__ += ["bert_score", "infolm"]
