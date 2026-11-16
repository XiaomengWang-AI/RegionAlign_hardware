"""Official VQA answer normalization and consensus accuracy.

The definitions below are the VQA scoring functions RegionAlign evaluates
LLaVA and Qwen2-VL with.  They follow the official VQAv2/TextVQA EvalAI
evaluators and have no dependency outside the standard library.
"""
from __future__ import annotations

import functools
import re

_VQA_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    "somethingll": "something'll", "thats": "that's", "thered": "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're",
    "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've",
    "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_VQA_NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
_VQA_ARTICLES = {"a", "an", "the"}
_VQA_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_VQA_COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")
_VQA_PUNCTUATION = (
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
    "_", "-", ">", "<", "@", "`", ",", "?", "!",
)


def _vqa_process_punctuation(text):
    """Official VQA punctuation normalization."""
    text = str(text)
    out = text
    for punctuation in _VQA_PUNCTUATION:
        if (punctuation + " " in text or " " + punctuation in text
                or _VQA_COMMA_STRIP.search(text) is not None):
            out = out.replace(punctuation, "")
        else:
            out = out.replace(punctuation, " ")
    return _VQA_PERIOD_STRIP.sub("", out)


def _vqa_process_digit_article(text):
    words = []
    for word in str(text).lower().split():
        word = _VQA_NUMBER_MAP.get(word, word)
        if word not in _VQA_ARTICLES:
            words.append(_VQA_CONTRACTIONS.get(word, word))
    return " ".join(words)


@functools.lru_cache(maxsize=65536)
def _normalize_vqa_answer_cached(answer, textvqa):
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    if textvqa:
        answer = answer.lower().replace(",", "").replace("?", "").replace("'s", " 's")
    return _vqa_process_digit_article(_vqa_process_punctuation(answer))


def normalize_vqa_answer(answer, textvqa=False):
    """Normalize as the official VQAv2/TextVQA EvalAI evaluators do.

    TextVQA applies its extra ``word_tokenize`` preprocessing before the shared
    VQA punctuation/digit/article normalization. VQAv2 does not.
    """
    return _normalize_vqa_answer_cached(str(answer), bool(textvqa))


def vqa_score(pred, answers, task=None):
    """Official VQA consensus accuracy averaged over held-out annotators.

    For each human answer, that annotator is held out and agreement with the
    other annotators contributes ``min(matches / 3, 1)``. With ten answers,
    predictions matching 1/2/3/4+ annotators therefore score .3/.6/.9/1.0.
    """
    textvqa = str(task).lower() == "textvqa"
    prediction = normalize_vqa_answer(pred, textvqa=textvqa)
    ground_truth = [normalize_vqa_answer(answer, textvqa=textvqa) for answer in answers]
    if not ground_truth:
        return 0.0
    total_matches = sum(answer == prediction for answer in ground_truth)
    annotator_scores = [
        min((total_matches - int(answer == prediction)) / 3.0, 1.0)
        for answer in ground_truth
    ]
    return sum(annotator_scores) / len(annotator_scores)
