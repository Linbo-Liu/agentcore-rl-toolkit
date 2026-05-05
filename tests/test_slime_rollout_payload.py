"""Tests for the slime backend's agent-payload conversion.

``_sample_to_payload`` is the contract between slime's ``Sample`` object
and the agent's ``@rollout_entrypoint`` payload. Post the data-contract
change, the rule is: the agent receives ``Sample.metadata`` verbatim
(shallow-copied). These tests pin that rule so the contract can't
regress silently.
"""

from types import SimpleNamespace

from agentcore_rl_toolkit.backends.slime.integration.rollout import _sample_to_payload


def test_metadata_is_returned_verbatim():
    sample = SimpleNamespace(metadata={"task_id": "t1", "prompt": "hi"})
    assert _sample_to_payload(sample) == {"task_id": "t1", "prompt": "hi"}


def test_empty_metadata_returns_empty_dict():
    sample = SimpleNamespace(metadata={})
    assert _sample_to_payload(sample) == {}


def test_missing_metadata_attr_returns_empty_dict():
    sample = SimpleNamespace()
    assert _sample_to_payload(sample) == {}


def test_none_metadata_returns_empty_dict():
    sample = SimpleNamespace(metadata=None)
    assert _sample_to_payload(sample) == {}


def test_non_dict_metadata_returns_empty_dict():
    sample = SimpleNamespace(metadata="not a dict")
    assert _sample_to_payload(sample) == {}


def test_returned_dict_is_a_shallow_copy():
    """Mutations to the returned payload must not leak into Sample.metadata.

    _process_one_episode mutates ``Sample.metadata`` (e.g. injects
    ``task_metadata``) downstream; the agent's view must remain stable.
    """
    metadata = {"prompt": "hi", "answer": "42"}
    sample = SimpleNamespace(metadata=metadata)

    payload = _sample_to_payload(sample)
    payload["injected"] = True

    assert "injected" not in metadata


def test_prompt_and_label_on_sample_are_ignored():
    """Top-level sample.prompt / sample.label are no longer part of the payload.

    They exist on ``Sample`` for slime's own use (tokenization, logging).
    Post the data-contract change, only ``metadata`` reaches the agent.
    """
    sample = SimpleNamespace(
        prompt="slime-side prompt",
        label="slime-side label",
        metadata={"foo": "bar"},
    )
    assert _sample_to_payload(sample) == {"foo": "bar"}
