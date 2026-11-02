"""Task 4 Stage 1: the digit smoke test (D135) — the picture must reach the model through the tool-result path."""
from agent_fakes import ScriptedClient

from bamr.agent.smoke import passed, run_smoke


def test_smoke_pass_and_fail(tmp_path):
    ok = run_smoke(ScriptedClient([[("show_picture", {})], "The number drawn is 7."]), tmp_path / "a")
    assert ok["pass"] is True and ok["text"] == "The number drawn is 7." and ok["gen_ids"] == ["gen-1", "gen-2"]
    bad = run_smoke(ScriptedClient([[("show_picture", {})], "I cannot see an image."]), tmp_path / "b")
    assert bad["pass"] is False
    none = run_smoke(ScriptedClient(["Seven, I guess."]), tmp_path / "c")
    assert none["pass"] is False and "no function call" in none["reason"]


def test_passed_rule():
    assert passed("7") and passed("It shows a seven.") and passed("Picture for call 1: the digit 7.")
    assert not passed("I see 1 and 7") and not passed("It is a 4.") and not passed("")


def test_smoke_picture_after_tool(tmp_path):
    c = ScriptedClient([[("show_picture", {})], "7"])
    run_smoke(c, tmp_path)
    m = c.seen[1]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool", "user"]
    assert m[3]["content"] == '{"shown": true}'
    assert m[4]["content"][0]["text"] == "picture for call 1" and m[4]["content"][1]["type"] == "image_url"


def test_same_settings_survives_json():
    """Review I1: the smoke file stores quantizations as a JSON list; the run must still match it to the tuple."""
    import dataclasses
    import json

    from bamr.agent.openrouter import MODELS
    from bamr.agent.smoke import same_settings
    for cfg in MODELS.values():
        saved = json.loads(json.dumps(dataclasses.asdict(cfg)))
        assert same_settings(saved, cfg)
        assert not same_settings(saved, dataclasses.replace(cfg, effort="none"))


def test_panel_count_is_not_a_reading():
    """Stage 2 r03 smoke: "the three panels" describes the layout the model saw; it is not a second number."""
    assert passed("The number drawn in the picture is **7**. It appears in the middle of the three panels.")
    assert passed("Seven, in the middle of 3 panels.")
    assert not passed("I think it is 3 or 7.")
    assert not passed("The three is drawn here.")


def test_panel_count_with_a_word_between_is_not_a_reading():
    """Held-out smoke, 29 Oct: Haiku read 7 and described "the two outer panels"; one or two words may sit between
    the count and "panels". A wrong digit is still caught."""
    assert passed("The number drawn in the picture is **7**. It appears in the middle panel, and the two outer panels "
                  "are just noise.")
    assert passed("It is a 7; the 2 side panels are empty.")
    assert not passed("It is a 1; the two outer panels are just noise.")
    assert not passed("The two outer panels show a 3.")


def test_number_words_that_count_things_are_not_readings():
    """Held-out smoke, 29 Oct (second): "with noise in the other two" — a count with no "panels" after it. Only
    numerals and "seven" count as readings; a wrong reading in words still fails because no 7 is given."""
    assert passed("The number drawn in the picture is **7**. It appears in the middle of the three panels, with noise "
                  "in the other two.")
    assert passed("Seven; one panel holds it, the other two are noise.")
    assert not passed("The number is three.")
    assert not passed("It is 1, in one of the three panels.")
    assert not passed("Either 7 or 1.")
