from types import SimpleNamespace

from scripts.train_lora_sft import (
    ImageBalancedSFTDataset,
    WallClockCheckpointCallback,
    filter_dataset_to_image_id,
    select_image_diverse_rows,
    training_args_from_config,
)


def test_fixed_generation_rows_are_image_diverse() -> None:
    rows = [
        {"image_id": "a", "caption": "a1"},
        {"image_id": "a", "caption": "a2"},
        {"image_id": "b", "caption": "b1"},
        {"image_id": "c", "caption": "c1"},
    ]
    selected = select_image_diverse_rows(rows, limit=2)
    assert [(row["image_id"], row["caption"]) for row in selected] == [("a", "a1"), ("b", "b1")]


def test_debug_dataset_can_select_explicit_stress_image() -> None:
    dataset = SimpleNamespace(rows=[{"image_id": "short"}, {"image_id": "longest"}])
    filter_dataset_to_image_id(dataset, "longest")
    assert dataset.rows == [{"image_id": "longest"}]


def test_image_balanced_dataset_has_one_slot_per_image_and_varies_train_caption() -> None:
    class Dataset:
        rows = [
            {"image_id": "a", "caption": "a1"},
            {"image_id": "a", "caption": "a2"},
            {"image_id": "b", "caption": "b1"},
        ]

        def __getitem__(self, index):
            return self.rows[index]

    balanced = ImageBalancedSFTDataset(Dataset(), seed=42)

    assert len(balanced) == 2
    assert {balanced[0]["caption"] for _ in range(20)} == {"a1", "a2"}
    assert balanced[1]["caption"] == "b1"


def test_image_balanced_validation_is_fixed() -> None:
    class Dataset:
        rows = [
            {"image_id": "a", "caption": "first"},
            {"image_id": "a", "caption": "second"},
        ]

        def __getitem__(self, index):
            return self.rows[index]

    balanced = ImageBalancedSFTDataset(Dataset(), seed=42, randomize=False)

    assert [balanced[0]["caption"] for _ in range(5)] == ["first"] * 5


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_wall_clock_checkpoint_requests_save_every_twelve_hours() -> None:
    clock = FakeClock(100.0)
    callback = WallClockCheckpointCallback(interval_hours=12, clock=clock)
    state = SimpleNamespace(global_step=1)
    control = SimpleNamespace(should_save=False)

    callback.on_train_begin(None, state, control)
    clock.now = 100.0 + 12 * 60 * 60 - 1
    callback.on_step_end(None, state, control)
    assert control.should_save is False

    clock.now += 1
    callback.on_step_end(None, state, control)
    assert control.should_save is True

    control.should_save = False
    clock.now += 12 * 60 * 60
    callback.on_step_end(None, state, control)
    assert control.should_save is True


def test_wall_clock_checkpoint_skips_missed_deadlines_without_save_storm() -> None:
    clock = FakeClock(0.0)
    callback = WallClockCheckpointCallback(interval_hours=12, clock=clock)
    state = SimpleNamespace(global_step=10)
    control = SimpleNamespace(should_save=False)

    callback.on_train_begin(None, state, control)
    clock.now = 30 * 60 * 60
    callback.on_step_end(None, state, control)
    assert control.should_save is True

    control.should_save = False
    clock.now += 1
    callback.on_step_end(None, state, control)
    assert control.should_save is False


def test_wall_clock_checkpoint_disables_step_based_save_strategy(tmp_path) -> None:
    config = {
        "model": {"device_map": "auto"},
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "num_epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "logging_steps": 25,
            "eval_steps": 1000,
            "save_steps": 1000,
            "save_hours": 12,
            "save_total_limit": 3,
        },
        "output": {"output_dir": str(tmp_path)},
    }

    args = training_args_from_config(config)

    assert args.save_strategy.value == "no"
