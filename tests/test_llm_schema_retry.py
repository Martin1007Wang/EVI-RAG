import json

from src.llm.eval_llm import (
    _LLMRequest,
    _parse_and_validate_response,
    _resolve_prompt_spec,
    _resolve_schema_spec,
    _retry_schema_batch,
)


class DummyBackend:
    def __init__(self, outputs_per_call):
        self._outputs = list(outputs_per_call)

    def generate(self, messages_batch):
        if not self._outputs:
            return [""] * len(messages_batch)
        outputs = self._outputs.pop(0)
        assert len(outputs) == len(messages_batch)
        return outputs


def _valid_payload(answer="A"):
    return json.dumps(
        {
            "answer": answer,
            "evidence_trajectory_ids": [1],
            "abstain_reason": "",
            "best_guess": "",
            "justification": "ok",
        }
    )


def test_schema_retry_batch():
    llm_cfg = {
        "prompt": {
            "system": "You are a test model.",
            "answer_key": "answer",
            "answer_separator": " | ",
            "allow_empty": True,
        },
        "schema": {
            "enabled": True,
            "max_retries": 2,
            "allow_coerce": True,
        },
    }
    prompt_spec = _resolve_prompt_spec(llm_cfg)
    schema_spec = _resolve_schema_spec(llm_cfg, prompt_spec)

    requests = [
        _LLMRequest(sample_id="s1", question="q1", trajectories=[], messages=[{"role": "user", "content": "x"}]),
        _LLMRequest(sample_id="s2", question="q2", trajectories=[], messages=[{"role": "user", "content": "y"}]),
    ]
    responses = ["not json", "still bad"]
    parsed = [_parse_and_validate_response(r, prompt_spec, schema_spec) for r in responses]

    backend = DummyBackend(
        [
            [_valid_payload("A"), "bad again"],
            [_valid_payload("B")],
        ]
    )

    responses, parsed, retries = _retry_schema_batch(
        backend=backend,
        batch_items=requests,
        responses=responses,
        parsed_list=parsed,
        prompt_spec=prompt_spec,
        schema_spec=schema_spec,
    )

    assert retries == [1, 2]
    assert parsed[0].schema_valid is True
    assert parsed[1].schema_valid is True
    assert parsed[0].answer == "A"
    assert parsed[1].answer == "B"
