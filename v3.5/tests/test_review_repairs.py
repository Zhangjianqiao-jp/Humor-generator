from types import SimpleNamespace
import json

import pytest
import torch
from torch import nn

from humor_generator_v35.homer.official_prompts import parse_caption_blob
from humor_generator_v35.homer.caption_parsing import parse_caption_strict
from humor_generator_v35.training.public_bridge import PUBLIC_BRIDGE_PROMPT_TRACK, selected_path_text
from humor_generator_v35.training.selected_plan import encode_selected_plan, selected_plan_sha256
from humor_generator_v35.training.memory_safe import configure_frozen_receiver
from humor_generator_v35.training.cross_attention_bridge import ReceiverCrossAttentionTask
from humor_generator_v35.training.gates import evaluate_mechanism_gate
from humor_generator_v35.training.formal_bridge import PreparedExample
from humor_generator_v35.latent.cross_attention import ReceiverDrivenCrossAttentionBridge
from scripts.finalize_homer_primary_evaluation import validate_records, SYSTEMS, GROUPS
from scripts.generate_pretrained_bridge_comparison import validate_checkpoint, load_completed_keys
from scripts.aggregate_homer_primary_judgments import aggregate


@pytest.mark.parametrize("caption,explanation", [("##Caption", "##Explanation"), ("## Caption", "## Explanation"), ("Caption", "Explanation")])
def test_strict_parser_extracts_only_caption(caption, explanation):
    result = parse_caption_strict(f"{caption}: Joke.\n{explanation}: Reason.")
    assert (result.caption, result.explanation, result.status) == ("Joke.", "Reason.", "ok")


@pytest.mark.parametrize("raw", ["Unmarked joke.", "##Caption: Joke.", "##Caption: A\n##Caption: B\n##Explanation: C", "##Explanation: Reason.\n##Caption: Joke."])
def test_strict_parser_rejects_ambiguous_output(raw):
    assert parse_caption_strict(raw).status != "ok"


def test_refusal_is_not_a_caption_and_official_parser_is_preserved():
    raw = "I’m sorry, but I can't answer this question."
    assert parse_caption_strict(raw).status == "known_refusal"
    assert parse_caption_blob(raw) == (raw, raw)


def grid():
    return [dict(system_id=s, image_id=i, trial=t, official_reference_group=g,
                 reference_group=f"{g}/gt_caption{j+1}", candidate_count=5, winning_caption_count=2)
            for s in SYSTEMS for i in ("i1", "i2") for t in range(5) for g in GROUPS for j in range(5)]


@pytest.mark.parametrize("bad", [2.9, True, "2", -1, 6])
def test_validation_rejects_illegal_wins(bad):
    rows = grid()
    rows[0]["winning_caption_count"] = bad
    with pytest.raises(ValueError):
        validate_records(rows)


def test_validation_uses_external_population_and_fixed_trials():
    rows = grid()
    validate_records(rows, expected_images={"i1", "i2"})
    for incomplete in ([r for r in rows if r["trial"] != 4], [r for r in rows if r["image_id"] != "i2"]):
        with pytest.raises(ValueError, match="incomplete HOMER grid"):
            validate_records(incomplete, expected_images={"i1", "i2"})


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.eval().requires_grad_(False)
        self.checkpoint_disabled = False
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, inputs_embeds, **kwargs):
        return self.layers[0](inputs_embeds.cumsum(1))

    def gradient_checkpointing_disable(self):
        self.checkpoint_disabled = True


class EncoderBackend:
    def __init__(self):
        self.model = EncoderModel()
        self.inputs = []

    def encode(self, messages, *, add_generation_prompt):
        text = messages[0]["content"][0]["text"]
        self.inputs.append(text)
        ids = torch.tensor([[ord(c) for c in text]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def multimodal_embeddings_and_positions(self, encoded):
        ids = encoded["input_ids"]
        return ids.float().unsqueeze(-1).expand(-1, -1, 4), torch.arange(ids.shape[1])[None]


def context():
    c = SimpleNamespace(selected_conflict="office vs animal", selected_entities=("cat", "desk"),
                        selected_paths={"cat": ("cat", "boss"), "desk": ("desk", "jungle")},
                        context_prompt_track=PUBLIC_BRIDGE_PROMPT_TRACK, trace_path="unused-a")
    c.free_association_text = selected_path_text(c)
    return c


def test_selected_memory_changes_with_plan_but_not_early_trace():
    backend = EncoderBackend()
    c = context()
    first = encode_selected_plan(backend, c)
    original_hash = selected_plan_sha256(c)
    c.trace_path = "unused-b"
    second = encode_selected_plan(backend, c)
    assert selected_plan_sha256(c) == original_hash
    for name in first:
        torch.testing.assert_close(first[name].states, second[name].states, rtol=0, atol=0)
        assert not first[name].states.requires_grad
    c.selected_conflict = "ocean vs city"
    third = encode_selected_plan(backend, c)
    assert selected_plan_sha256(c) != original_hash
    assert not torch.equal(first["conflict"].states, third["conflict"].states)
    assert not backend.model.layers[0]._forward_hooks
    assert all("unused" not in text for text in backend.inputs)


def test_freeze_report_matches_runtime_policy():
    model = EncoderModel().train()
    report = configure_frozen_receiver(model, nn.Linear(4, 4))
    assert not model.training
    assert model.checkpoint_disabled
    assert not report.gradient_checkpointing and not report.use_cache
    assert report.policy_trainable == 0 and report.bridge_trainable > 0


def test_diagnostics_always_describe_matched_memory():
    class Task(ReceiverCrossAttentionTask):
        def _states(self, cluster):
            return 2

        def _logits(self, embeddings, mask, positions, targets, states):
            self.bridge.last_diagnostics = [SimpleNamespace(
                gate=0.1, attention_entropy=states * 11, relative_update_norm=states * 0.1,
                channel_weights=(0.2, 0.3, 0.5))]
            return self.theta.expand(1, 2, 3) * states

    task = Task.__new__(Task)
    task.theta = nn.Parameter(torch.tensor([0.2, 0.4, 0.8]))
    task.bridge = SimpleNamespace(last_diagnostics=[])
    task.loss_config = dict(margin=0.2, matched_shuffled_margin=0.5, caption_nll=1)
    prepared = (None, None, None, torch.tensor([[0, 1]]), None, 1)
    for backward in (False, True):
        result = task._forward_metrics(prepared, "donor", backward=backward)
        assert result["mean_attention_entropy"] == 11
        assert result["shuffled_mean_attention_entropy"] == 22


def test_mechanism_gate_never_passes_missing_evidence():
    thresholds = {"min_validation_gap": 0.02, "min_nll_improvement": 0.01}
    assert evaluate_mechanism_gate({"matched_minus_shuffled_logp": 0.1}, thresholds)["status"] == "not_evaluated"
    assert evaluate_mechanism_gate({"matched_minus_shuffled_logp": 0.001}, thresholds)["status"] == "failed"
    assert evaluate_mechanism_gate({"matched_minus_shuffled_logp": 0.1, "nll_improvement_over_no_memory": 0.02}, thresholds)["status"] == "passed"


def test_new_method_rejects_old_checkpoint_before_loading_weights(tmp_path):
    checkpoint = tmp_path / "best_bridge.pt"
    checkpoint.write_bytes(b"not loaded by provenance check")
    (tmp_path / "complete.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "current_homer_public_route": True, "policy_trainable_parameters": 0,
        "baseline": "receiver_cross_attention"}))
    with pytest.raises(ValueError, match="method contract mismatch"):
        validate_checkpoint(checkpoint, expected_contract={
            "memory_source": "selected_plan_frozen_encoding_v1",
            "training_target_policy": "marked_caption_open_assistant_v1"})


def test_resume_rejects_different_generation_contract(tmp_path):
    output = tmp_path / "generation.jsonl"
    output.write_text(json.dumps({"generation_contract": {"checkpoint_sha256": "old"}}) + "\n")
    with pytest.raises(ValueError, match="contract mismatch"):
        load_completed_keys(output, condition="latent_bridge", rows=[], schedule=[], split="test",
                            expected_contract={"checkpoint_sha256": "new"})


def test_strict_failures_stay_in_denominator_even_if_judge_says_a(tmp_path):
    rows = [dict(packet_id=f"p{ref}-{candidate}", condition="latent_bridge", image_id="i1", trial=0,
                 reference_group="1-9", reference_index=ref, candidate_index=candidate,
                 generation_seed=candidate, caption_parser="strict_caption_v1",
                 parse_status="invalid_markers", invalid_caption_policy="count_as_failure")
            for ref in range(5) for candidate in range(5)]
    mapping = tmp_path / "mapping.jsonl"
    judgments = tmp_path / "judgments.jsonl"
    mapping.write_text("".join(json.dumps(row) + "\n" for row in rows))
    judgments.write_text("".join(json.dumps({"packet_id": r["packet_id"], "decision": "A"}) + "\n" for r in rows))
    output = tmp_path / "records.jsonl"
    aggregate(mapping, judgments, output)
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(result) == 5
    assert all(r["candidate_count"] == 5 and r["winning_caption_count"] == 0 for r in result)


def test_selected_plan_training_path_backpropagates_without_eos_or_policy_gradients(tmp_path):
    class Receiver(EncoderModel):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(128, 4)
            self.head = nn.Linear(4, 128)
            self.requires_grad_(False)

        def forward(self, inputs_embeds, logits_to_keep=0, **kwargs):
            hidden = self.layers[0](inputs_embeds)
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:]
            return SimpleNamespace(logits=self.head(hidden))

    class Backend:
        def __init__(self):
            self.model = Receiver()

        def encode(self, messages, *, add_generation_prompt, continue_final_message=False):
            text = ""
            for message in messages:
                content = message["content"]
                if isinstance(content, list):
                    content = "".join(part["text"] for part in content)
                text += "<" + message["role"] + ">" + content
            if add_generation_prompt:
                text += "<assistant>"
            elif not continue_final_message:
                text += "~"  # toy EOS, outside the target caption alphabet
            ids = torch.tensor([[ord(c) % 128 for c in text]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def multimodal_embeddings_and_positions(self, encoded):
            ids = encoded["input_ids"]
            return self.model.embedding(ids).detach(), torch.arange(ids.shape[1])[None]

    backend = Backend()
    bridge = ReceiverDrivenCrossAttentionBridge(4, 4, layer_indices=[0], bottleneck_dim=4, heads=1)
    configure_frozen_receiver(backend.model, bridge)
    c, donor = context(), context()
    c.cluster_id, c.contest_number, c.image, c.image_sha256, c.standard_description = "c1", 1, "x", "hash", "Cat at desk."
    donor.selected_conflict = "office vs ocean"
    row = {name: getattr(c, name) for name in ("cluster_id", "contest_number", "image", "image_sha256", "standard_description")}
    row.update(caption="Work for treats.", row_id="r1")
    task = ReceiverCrossAttentionTask(
        backend, bridge, root=tmp_path, trace_index={}, context_index={"c1": c, "donor": donor},
        loss_config=dict(caption_nll=1.0, text_teacher_forward_kl=0.5, matched_shuffled_margin=0.5, margin=0.2, temperature=1),
        max_target_tokens=128, stage="caption", prompt_mode=PUBLIC_BRIDGE_PROMPT_TRACK,
        memory_source="selected_plan_frozen_encoding_v1", target_policy="marked_caption_open_assistant_v1")
    example = PreparedExample(row, {})
    prepared = task.prepare(example)
    target = "".join(chr(t) for t in prepared[3][0].tolist())
    assert target == "##Caption: Work for treats."
    metrics = task._forward_metrics(prepared, "donor", backward=True)
    assert metrics["teacher_kl"] >= -1e-6
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.parameters())
    assert all(p.grad is None for p in backend.model.parameters())
    validation = task.evaluate_example(example, "donor")
    assert "no_memory_caption_nll" in validation
    assert validation["nll_improvement_over_no_memory"] == pytest.approx(validation["no_memory_caption_nll"] - validation["caption_nll"])
    assert not backend.model.layers[0]._forward_hooks


def test_smoke_measures_selected_memory_not_legacy_trace_length():
    from scripts.real_trace_bridge_smoke import memory_token_count
    from humor_generator_v35.training.selected_plan import SELECTED_PLAN_POLICY
    backend = EncoderBackend()
    c = context()
    count = memory_token_count(
        {"cluster_id": "a"}, memory_source=SELECTED_PLAN_POLICY,
        backend=backend, context_index={"a": c}, traces={},
    )
    encoded = encode_selected_plan(backend, c)
    assert count == sum(item.states.shape[1] for item in encoded.values())
    c.selected_conflict += " with a much longer explanation"
    assert memory_token_count(
        {"cluster_id": "a"}, memory_source=SELECTED_PLAN_POLICY,
        backend=backend, context_index={"a": c}, traces={},
    ) > count


def test_smoke_stresses_both_axes_and_preserves_two_clusters():
    from scripts.real_trace_bridge_smoke import select_stress_rows
    rows = [dict(cluster_id="image", pixels=100, tokens=1),
            dict(cluster_id="plan", pixels=1, tokens=100),
            dict(cluster_id="other", pixels=2, tokens=2)]
    chosen = select_stress_rows(rows, lambda r: r["pixels"], lambda r: r["tokens"])
    assert [r["cluster_id"] for r in chosen] == ["image", "plan"]
    rows[0]["tokens"] = 200
    chosen = select_stress_rows(rows, lambda r: r["pixels"], lambda r: r["tokens"])
    assert len({r["cluster_id"] for r in chosen}) == 2
    assert chosen[0]["cluster_id"] == "image"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_smoke_rejects_nonfinite_sample_before_optimizer(bad):
    from scripts.real_trace_bridge_smoke import require_finite_metrics
    with pytest.raises(RuntimeError, match="sample first"):
        require_finite_metrics({"nll": bad}, "sample first")
