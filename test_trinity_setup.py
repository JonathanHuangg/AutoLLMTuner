#!/usr/bin/env python
"""
Verification script to test Trinity-Mini quantization setup.
This script checks that all config files are properly formatted and accessible.
"""

import yaml
import json
from pathlib import Path

def test_configs():
    """Test that all Trinity config files load correctly"""

    configs_dir = Path("configs")

    print("=" * 60)
    print("Testing Trinity-Mini Quantization Setup")
    print("=" * 60)

    # 1. Test trinity_units.yaml
    print("\n[1/4] Testing trinity_units.yaml...")
    try:
        with open(configs_dir / "trinity_units.yaml") as f:
            units_config = yaml.safe_load(f)
        num_units = len(units_config["units"])
        num_groups = len(units_config["groups"])
        num_choices = len(units_config["choices"])
        print(f"  ✓ Loaded successfully")
        print(f"  - Units: {num_units}")
        print(f"  - Groups: {num_groups}")
        print(f"  - Bit choices: {units_config['choices']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # 2. Test trinity_layer_spec.yaml
    print("\n[2/4] Testing trinity_layer_spec.yaml...")
    try:
        with open(configs_dir / "trinity_layer_spec.yaml") as f:
            layer_spec = yaml.safe_load(f)
        num_specs = len(layer_spec["units"])
        print(f"  ✓ Loaded successfully")
        print(f"  - Layer specifications: {num_specs}")

        # Show a sample spec
        sample_key = list(layer_spec["units"].keys())[0]
        sample_spec = layer_spec["units"][sample_key]
        print(f"  - Sample ({sample_key}):")
        print(f"    op_type: {sample_spec['ops'][0]['op_type']}")
        print(f"    shape: m={sample_spec['ops'][0]['m']}, n={sample_spec['ops'][0]['n']}, k={sample_spec['ops'][0]['k']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # 3. Test trinity_param_counts.json
    print("\n[3/4] Testing trinity_param_counts.json...")
    try:
        with open(configs_dir / "trinity_param_counts.json") as f:
            param_counts = json.load(f)
        num_params = len(param_counts)
        total_params = sum(param_counts.values())
        print(f"  ✓ Loaded successfully")
        print(f"  - Parameter entries: {num_params}")
        print(f"  - Total quantizable params: {total_params:,} ({total_params / 1e9:.2f}B)")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # 4. Test trinity_run_config.yaml
    print("\n[4/4] Testing trinity_run_config.yaml...")
    try:
        with open(configs_dir / "trinity_run_config.yaml") as f:
            run_config = yaml.safe_load(f)
        print(f"  ✓ Loaded successfully")
        print(f"  - Run ID: {run_config['run_id']}")
        print(f"  - Model: {run_config['model']['name']}")
        print(f"  - Population size: {run_config['run']['population_size']}")
        print(f"  - Generations: {run_config['run']['generations']}")
        print(f"  - Mutation rate: {run_config['run']['mutation_rate']}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

    # 5. Verify consistency
    print("\n[5/5] Verifying consistency...")
    if num_units != num_specs:
        print(f"  ✗ Mismatch: {num_units} units vs {num_specs} specs")
        return False
    if num_units != num_params:
        print(f"  ✗ Mismatch: {num_units} units vs {num_params} param entries")
        return False
    print(f"  ✓ All counts match ({num_units} units)")

    print("\n" + "=" * 60)
    print("✓ All configuration files are valid!")
    print("=" * 60)

    print("\n📋 Next Steps:")
    print("  1. Verify bnb_quantize.py uses MODEL_NAME = 'arcee-ai/Trinity-Mini'")
    print("  2. Run evolutionary search:")
    print("     python -m src.evolutionary.main_search \\")
    print("       --config configs/trinity_run_config.yaml \\")
    print("       --run-id trinity-mini-test-001")
    print("\n")

    return True


if __name__ == "__main__":
    success = test_configs()
    exit(0 if success else 1)
