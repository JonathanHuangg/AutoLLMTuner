"""GroupTies: the constraint engine that keeps Q/K/V tied and bits legal."""

import yaml

from evolutionary.group_ties import GroupTies

SPEC = {
    "units": [
        {"name": "layers.0.self_attn.q_proj"},
        {"name": "layers.0.self_attn.k_proj"},
        {"name": "layers.0.self_attn.v_proj"},
        {"name": "layers.0.mlp.up_proj"},
        {"name": "layers.1.self_attn.q_proj"},
        {"name": "layers.1.self_attn.k_proj"},
        {"name": "layers.1.self_attn.v_proj"},
        {"name": "layers.1.mlp.up_proj"},
    ],
    "groups": [
        ["layers.*.self_attn.q_proj", "layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj"]
    ],
    "exclude": ["layers.1.mlp.up_proj"],
    "choices": [4, 8, 16],
}

NAMES = [u["name"] for u in SPEC["units"]]


def _ties(tmp_path):
    p = tmp_path / "units.yaml"
    p.write_text(yaml.safe_dump(SPEC))
    return GroupTies(NAMES, str(p))


def test_repair_ties_qkv_together(tmp_path):
    ties = _ties(tmp_path)
    # q/k/v are one gene across ALL layers: the group's representative is
    # the first index, so every q/k/v takes q's value (4) here.
    out = ties.repair((4, 8, 16, 8, 16, 4, 8, 8))
    assert out[0] == out[1] == out[2] == out[4] == out[5] == out[6] == 4
    # untied units are untouched
    assert out[3] == 8
    assert out[7] == 8


def test_repair_clamps_invalid_bits_to_nearest_choice(tmp_path):
    ties = _ties(tmp_path)
    out = ties.repair((2, 2, 2, 3, 6, 6, 6, 8))
    assert all(b in (4, 8, 16) for b in out)


def test_excluded_mask_marks_excluded_units(tmp_path):
    ties = _ties(tmp_path)
    assert ties.excluded_mask == [False] * 7 + [True]


def test_real_qwen_spec_ties_qkv_across_all_layers():
    from pathlib import Path

    from evolutionary.config import load_config

    cfg = load_config(
        str(Path(__file__).resolve().parents[1] / "configs" / "qwen2.5_run_config.yaml")
    )
    import yaml as _yaml

    names = [u["name"] for u in _yaml.safe_load(open(cfg.units.spec_file))["units"]]
    ties = GroupTies(names, cfg.units.spec_file)
    assert len(names) == 196
    repaired = ties.repair(tuple(4 if i % 3 == 0 else 8 for i in range(196)))
    qkv = [i for i, n in enumerate(names) if "self_attn.q_proj" in n]
    for i in qkv:
        layer = names[i].split(".")[1]
        k = names.index(f"layers.{layer}.self_attn.k_proj")
        v = names.index(f"layers.{layer}.self_attn.v_proj")
        assert repaired[i] == repaired[k] == repaired[v]
