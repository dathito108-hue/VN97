import pytest

from vn97 import (
    VN97ChatMessage,
    VN97DatasetSplitError,
    dataset_record_fingerprints,
    require_disjoint_dataset_splits,
)


def test_dataset_record_fingerprints_are_stable_for_text():
    first = dataset_record_fingerprints(
        ["alpha", "beta"],
        mode="text",
    )
    second = dataset_record_fingerprints(
        ["beta", "alpha"],
        mode="text",
    )
    assert first == second
    assert len(first) == 2


def test_three_way_split_rejects_text_record_overlap():
    with pytest.raises(VN97DatasetSplitError, match="training/release"):
        require_disjoint_dataset_splits(
            ["train only", "copied record"],
            training_mode="text",
            validation_records=["validation only"],
            validation_mode="text",
            release_records=["copied record", "release only"],
            release_mode="text",
        )


def test_three_way_split_rejects_chat_record_overlap():
    shared = (
        VN97ChatMessage("user", "same prompt"),
        VN97ChatMessage("assistant", "same answer"),
    )
    with pytest.raises(VN97DatasetSplitError, match="validation/release"):
        require_disjoint_dataset_splits(
            [
                (
                    VN97ChatMessage("user", "train"),
                    VN97ChatMessage("assistant", "train answer"),
                )
            ],
            training_mode="chat",
            validation_records=[shared],
            validation_mode="chat",
            release_records=[shared],
            release_mode="chat",
        )


def test_three_way_split_accepts_disjoint_records():
    require_disjoint_dataset_splits(
        ["training"],
        training_mode="text",
        validation_records=["validation"],
        validation_mode="text",
        release_records=["release"],
        release_mode="text",
    )
