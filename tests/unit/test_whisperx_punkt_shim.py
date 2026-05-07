from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer


def test_punkt_shim_keeps_common_abbreviations_with_sentence() -> None:
    params = PunktParameters()
    params.abbrev_types = {"dr", "vs", "mr", "mrs", "prof"}
    tokenizer = PunktSentenceTokenizer(params)

    text = "Dr. Maple joined. The meeting ended?"

    spans = list(tokenizer.span_tokenize(text))

    assert [text[start:end] for start, end in spans] == [
        "Dr. Maple joined.",
        "The meeting ended?",
    ]


def test_punkt_shim_supports_cjk_terminators() -> None:
    tokenizer = PunktSentenceTokenizer(PunktParameters())

    text = "第一句。第二句！"

    spans = list(tokenizer.span_tokenize(text))

    assert [text[start:end] for start, end in spans] == ["第一句。", "第二句！"]
