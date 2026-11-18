"""Regression tests for merging processing pipelines with ``+``.

``ProcessingPipeline.__add__`` originally reused the very same item objects in the merged
pipeline and cleared the pipeline binding of both operands. This broke three things:

* Processing items of the operands lost their pipeline relation. Transformations that store
  their result on the pipeline (e.g. :class:`SetStateTransformation`) silently wrote nowhere
  visible, because the transformation no longer knew the pipeline the operand was executed
  with.
* Merging one base pipeline with two different pipelines put the base item objects into both
  products. Binding a product re-bound the base items to that product, so the two products'
  states leaked into each other.
* A pipeline returned by :class:`ProcessingPipelineResolver` is the registered object itself.
  Merging it therefore corrupted the resolver's registry entry.

The merged pipeline must therefore contain copies that keep value equality with the source
entries, while the operands keep ownership of their items and behave exactly as before.
"""

from sigma.processing.pipeline import ProcessingPipeline, ProcessingItem
from sigma.processing.resolver import ProcessingPipelineResolver
from sigma.processing.transformations import SetStateTransformation
from sigma.rule import SigmaRule


def make_rule() -> SigmaRule:
    return SigmaRule.from_dict(
        {
            "title": "Test",
            "logsource": {"category": "test"},
            "detection": {"sel": {"field": "value"}, "condition": "sel"},
        }
    )


def state_pipeline(key: str, val: str) -> ProcessingPipeline:
    return ProcessingPipeline(
        items=[ProcessingItem(SetStateTransformation(key=key, val=val))],
    )


def test_merge_keeps_source_items_bound_to_their_pipeline():
    """Merging must not detach the operands' items from their own pipeline."""
    base = state_pipeline("base", "1")
    other = state_pipeline("other", "2")

    _ = base + other

    assert base.items[0]._pipeline is base
    assert base.items[0].transformation._pipeline is base
    assert other.items[0]._pipeline is other
    assert other.items[0].transformation._pipeline is other


def test_merged_pipeline_holds_copies_with_value_equality():
    """The merged pipeline contains copies, but value equality of the entries is preserved."""
    base = state_pipeline("base", "1")
    other = state_pipeline("other", "2")

    merged = base + other

    assert merged.items == base.items + other.items
    assert merged.items[0] is not base.items[0]
    assert merged.items[1] is not other.items[1]


def test_source_pipeline_executes_as_before_merge():
    """After a merge the source pipeline writes state to itself, not to the merge product."""
    base = state_pipeline("key", "value")
    merged = base + state_pipeline("other", "x")

    base.apply(make_rule())

    assert base.state == {"key": "value"}
    assert merged.state == {}


def test_merge_products_of_one_base_do_not_cross_talk():
    """One base merged with two pipelines yields products with independent state."""
    base = state_pipeline("base", "1")
    product_a = base + state_pipeline("a", "a")
    product_b = base + state_pipeline("b", "b")

    rule = make_rule()
    product_a.apply(rule)
    product_b.apply(rule)

    assert product_a.state == {"base": "1", "a": "a"}
    assert product_b.state == {"base": "1", "b": "b"}


def test_resolver_roundtrip_keeps_registered_pipeline_intact():
    """Merging a resolved pipeline must not modify the resolver's registry entry."""
    registered = state_pipeline("registered", "1")
    registered.name = "registered"
    resolver = ProcessingPipelineResolver({"registered": registered})

    merged = resolver.resolve(["registered"]) + state_pipeline("extra", "2")

    assert resolver.pipelines["registered"] is registered
    # The merged pipeline has an equal copy, not the registry object.
    assert merged.items[0] == registered.items[0]
    assert merged.items[0] is not registered.items[0]

    registered.apply(make_rule())

    assert registered.items[0]._pipeline is registered
    assert registered.state == {"registered": "1"}
    assert merged.state == {}
