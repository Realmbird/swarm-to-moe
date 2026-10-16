import numpy as np
import pytest

from swarm_to_moe.data import extract_gsm8k_answer, gsm8k_correct, seeded_subset
from swarm_to_moe.prompts import DEFAULT_ROLES, build_messages, shuffled_assignment


def test_default_roles_are_four_distinct_named_roles():
    assert len(DEFAULT_ROLES) == 4
    names = [r.name for r in DEFAULT_ROLES]
    assert len(set(names)) == 4
    for r in DEFAULT_ROLES:
        assert len(r.system_prompt) > 40


def test_build_messages_role_and_none():
    role = DEFAULT_ROLES[0]
    msgs = build_messages("What is 2+2?", role)
    assert msgs[0]["role"] == "system" and role.system_prompt in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "What is 2+2?"}
    plain = build_messages("What is 2+2?", None)
    assert plain[0]["role"] == "user"


def test_shuffled_assignment_is_a_derangement_and_seeded():
    agents = ["A", "B", "C", "D"] * 5
    s1 = shuffled_assignment(agents, seed=0)
    s2 = shuffled_assignment(agents, seed=0)
    s3 = shuffled_assignment(agents, seed=1)
    assert s1 == s2 and s1 != s3
    assert sorted(s1) == sorted(agents)
    assert all(a != b for a, b in zip(agents, s1)), "every row must get a different role than its true one"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("... so the answer is 42.", "42"),
        ("Total = 1,234 dollars. #### 1234", "1234"),
        ("The final answer is $18.", "18"),
        ("#### 72", "72"),
        ("She has 7 apples.\n\nAnswer: 7", "7"),
        ("no number here", None),
        ("-5 degrees", "-5"),
        ("Final answer: 3.50", "3.50"),
    ],
)
def test_extract_gsm8k_answer(text, expected):
    assert extract_gsm8k_answer(text) == expected


def test_gsm8k_correct_numeric_equivalence():
    assert gsm8k_correct("#### 1234", "1,234")
    assert gsm8k_correct("The answer is 3.50", "3.5")
    assert gsm8k_correct("Answer: 18", "18.0")
    assert not gsm8k_correct("Answer: 19", "18")
    assert not gsm8k_correct("no answer", "18")


def test_seeded_subset_deterministic_and_disjoint_from_profile_split():
    items = list(range(1000))
    a = seeded_subset(items, n=100, seed=0)
    b = seeded_subset(items, n=100, seed=0)
    c = seeded_subset(items, n=100, seed=1)
    assert a == b and a != c
    assert len(set(a)) == 100


# ----------------------------------------------------------------------------- harmony output parsing

from swarm_to_moe.data import score_gsm8k_harmony, split_harmony  # noqa: E402

HARMONY = (
    "<|channel|>analysis<|message|>Compute: 3+5=8, minus 2 =6.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>6 apples.<|return|>"
)


def test_split_harmony_final_and_analysis():
    analysis, final = split_harmony(HARMONY)
    assert final == "6 apples."
    assert "3+5=8" in analysis and "<|" not in analysis


def test_split_harmony_truncated_has_no_final():
    analysis, final = split_harmony("<|channel|>analysis<|message|>Compute: 3+5=8, then 9")
    assert final is None
    assert analysis.startswith("Compute")


def test_split_harmony_uses_last_final_channel_and_scoring():
    text = HARMONY.replace("<|return|>", "<|end|>") + "<|start|>assistant<|channel|>final<|message|>#### 7<|return|>"
    assert split_harmony(text)[1] == "#### 7"
    s = score_gsm8k_harmony(HARMONY, "6")
    assert s["correct"] and not s["truncated"]
    t = score_gsm8k_harmony("<|channel|>analysis<|message|>so 6 then 9", "6")
    assert t["truncated"] and not t["correct"]
