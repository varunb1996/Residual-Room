from eval_cases import case, eval_cases

from residual_room.checker.claims import count_ctas, split_segments


def kinds(segments):
    return [(s.kind, s.exempt_type) for s in segments]


def test_all_50_eval_cases_load():
    assert len(eval_cases()) == 50


def test_cln01_email_greeting_claim_cta_signoff():
    segs = split_segments(case("CLN-01")["text"])
    assert kinds(segs) == [
        ("exempt", "greeting"),
        ("candidate", None),  # "This week in First Principles: ..." (LLM decides)
        ("candidate", None),
        ("exempt", "cta"),
        ("exempt", "sign_off"),
    ]
    assert segs[2].citations == ("ext-rag-paper",)
    assert "[cite:" not in segs[2].text and segs[2].text.endswith("Wikipedia.")


def test_cln04_marked_opinion_is_exempt():
    segs = split_segments(case("CLN-04")["text"])
    # Line 1 is two sentences: the series label, then the cited claim.
    assert kinds(segs) == [
        ("candidate", None),
        ("candidate", None),
        ("exempt", "opinion"),
        ("exempt", "cta"),
    ]
    assert segs[1].citations == ("ext-mtbench-paper",)


def test_cln08_read_next_is_not_a_cta():
    segs = split_segments(case("CLN-08")["text"])
    assert segs[-1].exempt_type == "read_next"
    assert count_ctas(case("CLN-08")["text"]) == 0


def test_pol10_counts_three_ctas_across_two_sentences():
    assert count_ctas(case("POL-10")["text"]) == 3
    assert {s.exempt_type for s in split_segments(case("POL-10")["text"])} == {"cta"}


def test_numbers_and_multiple_citations_stay_in_one_sentence():
    segs = split_segments(case("CLN-05")["text"])
    assert segs[-1].citations == ("exp-003-v2", "int-corrections")
    (seg,) = split_segments(case("CLN-02")["text"])[1:2]
    assert "23.8%" in seg.text and "65.0%" in seg.text


def test_unsubscribe_is_not_a_subscribe_cta():
    assert count_ctas("Every email has an unsubscribe link.") == 0
