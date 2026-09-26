from __future__ import annotations

import hashlib

from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    VN97P4DRecord,
    build_p4d_curriculum,
    default_training as p4d_training,
    default_validation as p4d_validation,
)


P4E_C_PROFILE_ID = "vn97-p4e-c-targeted-generation-repair-v1"
P4E_C_TRAIN_PER_CATEGORY = 600
P4E_C_TRAIN_SEED = 51997


def targeted_training() -> tuple[VN97P4DRecord, ...]:
    """Build a deterministic repair set disjoint from all P4D train/validation prompts."""
    blocked = {
        item.prompt
        for item in (
            *p4d_training(),
            *p4d_validation(),
        )
    }
    candidates = build_p4d_curriculum(
        validation=False,
        per_category=P4E_C_TRAIN_PER_CATEGORY * 3,
        seed=P4E_C_TRAIN_SEED,
    )
    selected: list[VN97P4DRecord] = []
    counts = {
        category: 0
        for category in P4D_CATEGORIES
    }

    ranked = sorted(
        candidates,
        key=lambda item: (
            item.category,
            hashlib.sha256(
                item.prompt.encode("utf-8")
            ).digest(),
        ),
    )
    seen: set[str] = set()
    for item in ranked:
        if (
            item.prompt in blocked
            or item.prompt in seen
            or counts[item.category]
            >= P4E_C_TRAIN_PER_CATEGORY
        ):
            continue
        selected.append(item)
        seen.add(item.prompt)
        counts[item.category] += 1

    expected = {
        category: P4E_C_TRAIN_PER_CATEGORY
        for category in P4D_CATEGORIES
    }
    if counts != expected:
        raise RuntimeError(
            "could not build balanced disjoint P4E-C repair curriculum"
        )

    selected.sort(
        key=lambda item: hashlib.sha256(
            (
                item.category
                + "\0"
                + item.prompt
            ).encode("utf-8")
        ).digest()
    )
    return tuple(selected)
