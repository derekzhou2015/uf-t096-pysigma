"""Regression tests for processing pipeline merging.

Merging two pipelines must produce an independent merged pipeline without mutating
the operands or the items they own. Previously ``__add__`` detached the operand
pipelines' items and rebound them to the merged pipeline, which caused:

* state-setting transformations executed through an operand pipeline after the
  merge to write their state into the merged pipeline (silently lost on the operand);
* repeated merges of the same base pipeline to cross-contain each other's state;
* pipelines registered in a resolver to be corrupted by later merge operations.
"""

import pytest

from sigma.processing.pipeline import ProcessingPipeline, ProcessingItem
from sigma.processing.resolver import ProcessingPipelineResolver
from sigma.processing.transformations import SetStateTransformation
from sigma.rule import SigmaRule


@pytest.fixture
def sigma_rule():
    return SigmaRule.from_dict(
        {
            "title": "Test",
            "logsource": {"category": "test"},
            "detection": {
                "test": {"field": "value"},
                "condition": "test",
            },
        }
    )


def state_pipeline(name: str, key: str, val) -> ProcessingPipeline:
    return ProcessingPipeline(
        name=name,
        items=[
            ProcessingItem(
                identifier=f"set-{key}",
                transformation=SetStateTransformation(key=key, val=val),
            )
        ],
    )


def test_merge_does_not_mutate_operand_pipeline(sigma_rule):
    """An operand pipeline keeps owning its items and its execution results after merge."""
    base = state_pipeline("base", "k", "base")
    other = state_pipeline("other", "j", "other")
    original_item = base.items[0]

    merged = base + other

    # Ownership of the original item tree stays with the operand pipeline.
    assert base.items[0] is original_item
    assert original_item._pipeline is base
    assert original_item.transformation._pipeline is base

    # Execution on the operand behaves exactly as if the merge never happened.
    base.apply(sigma_rule)
    assert base.state == {"k": "base"}
    assert merged.state == {}


def test_merge_operand_execution_independent_of_merged_execution(sigma_rule):
    """State written on an operand must never land on the merged pipeline and vice versa."""
    base = state_pipeline("base", "k", "base")
    other = state_pipeline("other", "j", "other")
    merged = base + other

    merged.apply(sigma_rule)
    assert merged.state == {"k": "base", "j": "other"}
    assert base.state == {}

    base.apply(sigma_rule)
    assert base.state == {"k": "base"}
    assert merged.state == {"k": "base", "j": "other"}


def test_repeated_merge_of_same_base_does_not_cross_contaminate(sigma_rule):
    """Two merge products sharing one base pipeline hold independent state."""
    base = state_pipeline("base", "k", "base")
    extra_a = state_pipeline("extra-a", "a", "a-val")
    extra_b = state_pipeline("extra-b", "b", "b-val")

    merged_a = base + extra_a
    merged_b = base + extra_b

    merged_a.apply(sigma_rule)
    assert merged_a.state == {"k": "base", "a": "a-val"}

    merged_b.apply(sigma_rule)
    assert merged_b.state == {"k": "base", "b": "b-val"}
    # The first merge product is untouched by the second execution.
    assert merged_a.state == {"k": "base", "a": "a-val"}


def test_merge_uses_independent_item_copies():
    """The merged pipeline gets its own item objects instead of stealing the operands'."""
    base = state_pipeline("base", "k", "base")
    other = state_pipeline("other", "j", "other")

    merged = base + other

    assert merged.items[0] is not base.items[0]
    assert merged.items[1] is not other.items[0]
    assert merged.items[0]._pipeline is merged
    assert merged.items[1]._pipeline is merged
    # Value equality semantics are preserved.
    assert merged.items == base.items + other.items
    assert merged == ProcessingPipeline(
        items=[
            ProcessingItem(
                identifier="set-k",
                transformation=SetStateTransformation(key="k", val="base"),
            ),
            ProcessingItem(
                identifier="set-j",
                transformation=SetStateTransformation(key="j", val="other"),
            ),
        ]
    )


def test_resolver_registered_pipeline_unaffected_by_resolution(sigma_rule):
    """Merging through a resolver must not corrupt the pipeline object in the registry."""
    base = state_pipeline("base", "k", "base")
    other = state_pipeline("other", "j", "other")
    resolver = ProcessingPipelineResolver.from_pipeline_list([base, other])

    resolved = resolver.resolve(["base", "other"])
    resolved.apply(sigma_rule)
    assert resolved.state == {"k": "base", "j": "other"}

    # The registered pipeline resolves to the same object and still works on its own.
    registered = resolver.resolve_pipeline("base")
    assert registered is base
    assert base.items[0]._pipeline is base
    base.apply(sigma_rule)
    assert base.state == {"k": "base"}
    assert resolved.state == {"k": "base", "j": "other"}


def test_resolver_round_trip_still_merges_values(sigma_rule):
    """Resolver round trip keeps the concatenation semantics after the isolation fix."""
    base = state_pipeline("base", "k", "base")
    other = state_pipeline("other", "j", "other")
    resolver = ProcessingPipelineResolver.from_pipeline_list([base, other])

    first = resolver.resolve(["base", "other"])
    second = resolver.resolve(["base", "other"])

    assert first is not second
    assert first == second
    second.apply(sigma_rule)
    assert second.state == {"k": "base", "j": "other"}
    assert first.state == {}
