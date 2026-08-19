"""Smoke test for cache_injector, using synthetic caches (no model load needed)."""

import torch
from transformers.cache_utils import DynamicCache

from cache_injector import AppendConfig, append_kv_cache


def _make_cache(num_layers: int, seq_len: int, fill_value: float) -> DynamicCache:
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        keys = torch.full((1, 2, seq_len, 4), fill_value)
        values = torch.full((1, 2, seq_len, 4), fill_value)
        cache.update(keys, values, layer_idx)
    return cache


def test_append_extends_every_layer_at_the_tail() -> None:
    active = _make_cache(num_layers=3, seq_len=50, fill_value=1.0)
    shadow = _make_cache(num_layers=3, seq_len=20, fill_value=9.0)
    config = AppendConfig(append_percent=0.5, layers_affect_percent=1.0)

    result = append_kv_cache(active, shadow, config)

    assert result.active_len_before == 50
    assert result.num_positions_appended == 10  # round(0.5 * 20)
    assert result.active_len_after == 60

    for layer in active.layers:
        assert layer.keys.shape[2] == 60
        assert torch.all(layer.keys[:, :, :50, :] == 1.0)  # real content untouched
        assert torch.all(layer.keys[:, :, 50:60, :] == 9.0)  # appended phantom content
        assert torch.all(layer.values[:, :, 50:60, :] == 9.0)


def test_full_append_uses_entire_shadow_cache() -> None:
    active = _make_cache(num_layers=2, seq_len=30, fill_value=1.0)
    shadow = _make_cache(num_layers=2, seq_len=15, fill_value=9.0)
    config = AppendConfig(append_percent=1.0, layers_affect_percent=1.0)

    result = append_kv_cache(active, shadow, config)

    assert result.num_positions_appended == 15
    assert result.active_len_after == 45
    layer0 = active.layers[0]
    assert torch.all(layer0.keys[:, :, 30:45, :] == 9.0)


def test_zero_append_is_a_noop() -> None:
    active = _make_cache(num_layers=2, seq_len=30, fill_value=1.0)
    shadow = _make_cache(num_layers=2, seq_len=15, fill_value=9.0)
    config = AppendConfig(append_percent=0.0, layers_affect_percent=1.0)

    result = append_kv_cache(active, shadow, config)

    assert result.num_positions_appended == 0
    assert result.active_len_after == 30
    assert active.layers[0].keys.shape[2] == 30


def test_partial_layers_rejected() -> None:
    active = _make_cache(num_layers=2, seq_len=30, fill_value=1.0)
    shadow = _make_cache(num_layers=2, seq_len=15, fill_value=9.0)
    config = AppendConfig(append_percent=0.5, layers_affect_percent=0.5)

    try:
        append_kv_cache(active, shadow, config)
        raise AssertionError("expected ValueError for layers_affect_percent != 1.0")
    except ValueError:
        pass


def test_mismatched_layer_counts_rejected() -> None:
    active = _make_cache(num_layers=3, seq_len=30, fill_value=1.0)
    shadow = _make_cache(num_layers=2, seq_len=15, fill_value=9.0)
    config = AppendConfig(append_percent=0.5, layers_affect_percent=1.0)

    try:
        append_kv_cache(active, shadow, config)
        raise AssertionError("expected ValueError for mismatched layer counts")
    except ValueError:
        pass


if __name__ == "__main__":
    test_append_extends_every_layer_at_the_tail()
    test_full_append_uses_entire_shadow_cache()
    test_zero_append_is_a_noop()
    test_partial_layers_rejected()
    test_mismatched_layer_counts_rejected()
    print("all cache_injector smoke tests passed")
