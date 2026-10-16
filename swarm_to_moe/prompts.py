"""Role system prompts (PLACEHOLDERS until Martin's four LoRA'd roles are known) and chat helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Role:
    name: str
    system_prompt: str


DEFAULT_ROLES: tuple[Role, ...] = (
    Role(
        "planner",
        "You are the PLANNER in a team of agents. Your job is to decompose the task into a short numbered plan "
        "of sub-steps and say which quantities each step needs. Never lead with the final answer; the plan "
        "comes first, and only after the plan may you state where it leads.",
    ),
    Role(
        "critic",
        "You are the CRITIC in a team of agents. Look for flaws, hidden assumptions, and likely mistakes in "
        "the reasoning a solver would use on this task. State every problem you find bluntly and briefly, "
        "without softening, then say what the correct handling should be.",
    ),
    Role(
        "implementer",
        "You are the IMPLEMENTER in a team of agents. Give the direct worked solution: carry out the steps, "
        "show the essential computations, and finish with the final answer. Do not discuss alternatives.",
    ),
    Role(
        "verifier",
        "You are the VERIFIER in a team of agents. Re-check every arithmetic operation and logical step one "
        "at a time, writing each check explicitly. Finish by stating the result and your confidence in it "
        "as a percentage.",
    ),
)

ROLE_BY_NAME = {r.name: r for r in DEFAULT_ROLES}


def build_messages(user_text: str, role: Role | None) -> list[dict]:
    msgs = []
    if role is not None:
        msgs.append({"role": "system", "content": role.system_prompt})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def shuffled_assignment(agents: Sequence[str], seed: int, max_tries: int = 1_000_000) -> list[str]:
    """Seeded permutation of ``agents`` such that no position keeps its original label (a derangement by
    value). Rejection-sampled with numpy default_rng(seed); preserves the multiset."""
    agents = list(agents)
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        perm = rng.permutation(len(agents))
        out = [agents[i] for i in perm]
        if all(a != b for a, b in zip(agents, out)):
            return out
    raise ValueError("could not find a derangement (is one label more than half of the list?)")
