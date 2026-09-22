import json
import math
from pathlib import Path

import pytest
import torch

from vn97.cognition import (
    CognitionLoop,
    LoopBoundary,
    MemoryQueryRequest,
    PlanDraftRequest,
    StepReasoningRequest,
    VerificationRequest,
)
from vn97.cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97CognitionAdapter,
    VN97CognitionOutputError,
    VN97InferenceContractError,
    VN97InferenceLimits,
)
from vn97.config import VN97Config
from vn97.memory import MemoryJournal, MemoryKind
from vn97.model import VN97LanguageCore
from vn97.planner import MemoryContext, ReasoningBudget, StepKind
from vn97.tokenizer import VN97Tokenizer


class ScriptedEngine:
    def __init__(self, outputs, *, vector=(1.0, 0.0)):
        self.outputs = list(outputs)
        self.vector = tuple(vector)
        self.prompts = []
        self.embed_calls = []

    def generate_text(self, prompt: str, *, max_new_tokens: int) -> str:
        self.prompts.append((prompt, max_new_tokens))
        if not self.outputs:
            raise RuntimeError("script exhausted")
        return self.outputs.pop(0)

    def embed_text(self, text: str, *, vector_dim: int) -> tuple[float, ...]:
        self.embed_calls.append((text, vector_dim))
        if len(self.vector) != vector_dim:
            raise VN97InferenceContractError("script embedding dimension mismatch")
        return self.vector


def _plan_json(*steps):
    return json.dumps({"steps": list(steps)}, separators=(",", ":"))


def _step(kind, objective, dependencies=(), verify=False, confidence=0.0):
    return {
        "kind": kind,
        "objective": objective,
        "dependencies": list(dependencies),
        "requires_verification": verify,
        "min_confidence": confidence,
    }


def test_adapter_parses_strict_plan_and_preserves_schema_prompt():
    engine = ScriptedEngine([
        _plan_json(
            _step("REASON", "analyze"),
            _step("RESPOND", "answer", (1,), True, 0.8),
        )
    ])
    adapter = VN97CognitionAdapter(engine)
    draft = adapter.propose_plan(PlanDraftRequest(goal="goal", max_steps=8))
    assert [step.kind for step in draft.steps] == [StepKind.REASON, StepKind.RESPOND]
    assert draft.steps[1].dependencies == (1,)
    assert draft.steps[1].requires_verification
    prompt, limit = engine.prompts[0]
    assert prompt.startswith("VN97COG1\noperation=plan\n")
    assert '"goal":"goal"' in prompt
    assert limit > 0


def test_adapter_rejects_markdown_extra_and_duplicate_keys():
    for output in (
        '\x60\x60\x60json\n{"result":"x","confidence":1.0}\n\x60\x60\x60',
        '{"result":"x","confidence":1.0,"extra":1}',
        '{"result":"x","result":"y","confidence":1.0}',
    ):
        adapter = VN97CognitionAdapter(ScriptedEngine([output]))
        request = StepReasoningRequest(
            plan_id="p",
            goal="g",
            step_id=1,
            kind=StepKind.REASON,
            objective="o",
            attempt=1,
            previous_failure="",
            dependencies=(),
            memory_context=MemoryContext(()),
            context_truncated=False,
        )
        with pytest.raises(VN97CognitionOutputError):
            adapter.propose_step(request)


def test_memory_query_embeds_model_selected_text_locally():
    output = json.dumps(
        {
            "query": "VN97 sovereign memory",
            "top_k": 2,
            "kinds": ["EPISODIC", "SEMANTIC"],
            "semantic_weight": 1.0,
            "recency_weight": 0.1,
            "importance_weight": 0.2,
            "recency_half_life_ns": 1000,
        },
        separators=(",", ":"),
    )
    engine = ScriptedEngine([output], vector=(0.8, 0.2))
    adapter = VN97CognitionAdapter(engine)
    query = adapter.memory_query(
        MemoryQueryRequest(
            plan_id="p",
            goal="g",
            step_id=1,
            objective="retrieve",
            attempt=1,
            previous_failure="",
            dependencies=(),
            vector_dim=2,
        )
    )
    assert query.vector == (0.8, 0.2)
    assert query.kinds == (MemoryKind.EPISODIC, MemoryKind.SEMANTIC)
    assert engine.embed_calls == [("VN97 sovereign memory", 2)]


def test_memory_query_dimension_contract_fails_closed():
    output = json.dumps(
        {
            "query": "x",
            "top_k": 1,
            "kinds": None,
            "semantic_weight": 1.0,
            "recency_weight": 0.0,
            "importance_weight": 0.0,
            "recency_half_life_ns": 10,
        }
    )
    adapter = VN97CognitionAdapter(ScriptedEngine([output], vector=(1.0, 0.0)))
    with pytest.raises(VN97CognitionOutputError, match="dimension"):
        adapter.memory_query(
            MemoryQueryRequest(
                plan_id="p",
                goal="g",
                step_id=1,
                objective="r",
                attempt=1,
                previous_failure="",
                dependencies=(),
                vector_dim=3,
            )
        )


def test_step_and_verification_outputs_are_typed():
    engine = ScriptedEngine([
        '{"result":"candidate","confidence":0.75}',
        '{"passed":true,"note":"supported"}',
    ])
    adapter = VN97CognitionAdapter(engine)
    proposal = adapter.propose_step(
        StepReasoningRequest(
            plan_id="p",
            goal="g",
            step_id=1,
            kind=StepKind.REASON,
            objective="o",
            attempt=1,
            previous_failure="",
            dependencies=(),
            memory_context=MemoryContext(()),
            context_truncated=False,
        )
    )
    assert proposal.result == "candidate"
    assert proposal.confidence == 0.75
    decision = adapter.verify_step(
        VerificationRequest(
            plan_id="p",
            goal="g",
            step_id=1,
            kind=StepKind.REASON,
            objective="o",
            attempt=1,
            candidate="candidate",
            confidence=0.75,
            dependencies=(),
            evidence_record_ids=(),
        )
    )
    assert decision.passed
    assert decision.note == "supported"


def test_adapter_runs_end_to_end_through_m5b_with_real_memory(tmp_path: Path):
    plan = _plan_json(
        _step("RETRIEVE", "find evidence"),
        _step("RESPOND", "answer", (1,)),
    )
    query = json.dumps(
        {
            "query": "memory evidence",
            "top_k": 1,
            "kinds": None,
            "semantic_weight": 1.0,
            "recency_weight": 0.0,
            "importance_weight": 0.0,
            "recency_half_life_ns": 100,
        },
        separators=(",", ":"),
    )
    engine = ScriptedEngine(
        [
            plan,
            query,
            '{"result":"memory synthesis","confidence":1.0}',
            '{"result":"final from VN97","confidence":1.0}',
        ],
        vector=(1.0, 0.0),
    )
    adapter = VN97CognitionAdapter(engine)
    loop = CognitionLoop(adapter)
    controller = loop.build_plan(
        "answer from memory",
        budget=ReasoningBudget(max_transitions=20, max_memory_queries=2),
    )
    journal = MemoryJournal.create(tmp_path / "m.vn97mem", vector_dim=2)
    record = journal.append(
        MemoryKind.SEMANTIC,
        "trusted fact",
        source="unit",
        vector=[1.0, 0.0],
        timestamp_ns=1,
    )
    result = loop.run_until_boundary(controller, journal=journal)
    assert result.boundary == LoopBoundary.COMPLETED
    assert result.final_response == "final from VN97"
    assert controller.plan.step(1).evidence_record_ids == (record.record_id,)
    assert controller.plan.step(2).evidence_record_ids == (record.record_id,)
    assert [p[0].splitlines()[1] for p in engine.prompts] == [
        "operation=plan",
        "operation=memory_query",
        "operation=step",
        "operation=step",
    ]


def test_torch_engine_hidden_projection_matches_forward_logits():
    torch.manual_seed(91)
    tokenizer = VN97Tokenizer()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=12,
            n_layers=1,
            d_state=4,
        )
    ).eval()
    ids = torch.tensor([tokenizer.encode("hello", add_bos=True, add_text_tag=True)])
    with torch.no_grad():
        hidden, hidden_states = model.forward_hidden(ids)
        logits, states = model(ids)
        expected = model.lm_head(hidden)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert len(hidden_states) == len(states) == 1


def test_torch_engine_produces_native_hidden_retrieval_vector():
    torch.manual_seed(92)
    tokenizer = VN97Tokenizer()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=12,
            n_layers=1,
            d_state=4,
        )
    )
    engine = TorchVN97InferenceEngine(model, tokenizer)
    vector = engine.embed_text("VN97 memory", vector_dim=12)
    assert len(vector) == 12
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) > 0.0
    with pytest.raises(VN97InferenceContractError, match="vector_dim"):
        engine.embed_text("VN97 memory", vector_dim=8)


def test_torch_engine_vocab_and_prompt_limits_fail_closed():
    tokenizer = VN97Tokenizer()
    mismatch = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size + 1,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    with pytest.raises(VN97InferenceContractError, match="vocab_size"):
        TorchVN97InferenceEngine(mismatch, tokenizer)

    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(max_prompt_tokens=3),
    )
    with pytest.raises(VN97InferenceContractError, match="prompt"):
        engine.generate_text("long prompt", max_new_tokens=1)


def test_torch_engine_generation_is_bounded_and_uses_recurrent_path():
    tokenizer = VN97Tokenizer()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    engine = TorchVN97InferenceEngine(model, tokenizer)
    assert engine.generate_text("x", max_new_tokens=3) == ""
