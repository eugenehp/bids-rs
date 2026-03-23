#!/usr/bin/env python3
"""
Detect new functions, modules, datatypes, and entities in PyBIDS
that bids-rs doesn't yet cover.

Exit code 0: no new changes
Exit code 1: new changes detected (CI should alert)
"""

import inspect
import importlib
import sys
import json
from pathlib import Path

KNOWN_FILE = Path(__file__).parent / "golden" / "known_pybids_api.json"

alerts = []
info = []

def extract_public_api():
    """Extract all public classes, functions, and methods from PyBIDS."""
    api = {}

    modules = [
        "bids",
        "bids.layout",
        "bids.layout.layout",
        "bids.layout.models",
        "bids.layout.writing",
        "bids.layout.utils",
        "bids.layout.validation",
        "bids.layout.index",
        "bids.layout.db",
        "bids.variables",
        "bids.variables.variables",
        "bids.variables.collections",
        "bids.variables.io",
        "bids.variables.entities",
        "bids.modeling",
        "bids.modeling.statsmodels",
        "bids.modeling.hrf",
        "bids.modeling.auto_model",
        "bids.modeling.model_spec",
        "bids.modeling.transformations",
        "bids.modeling.transformations.base",
        "bids.modeling.transformations.compute",
        "bids.modeling.transformations.munge",
        "bids.reports",
        "bids.reports.report",
        "bids.reports.parsing",
        "bids.reports.parameters",
        "bids.utils",
        "bids.config",
        "bids.cli",
    ]

    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue

        members = {}
        for name, obj in inspect.getmembers(mod):
            if name.startswith("_"):
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if inspect.isclass(obj):
                    methods = [m for m in dir(obj) if not m.startswith("_") and callable(getattr(obj, m, None))]
                    members[name] = {"type": "class", "methods": sorted(methods)}
                else:
                    members[name] = {"type": "function"}

        if members:
            api[mod_name] = members

    return api


def extract_entities_and_datatypes():
    """Extract current entity names and datatypes from PyBIDS config."""
    result = {}

    try:
        from bids.layout.models import Config
        config = Config.load("bids")
        result["entities"] = sorted(config.entities.keys())
    except Exception:
        result["entities"] = []

    try:
        from bids.layout.models import Config
        config = Config.load("bids")
        # Extract datatypes from entity pattern
        for ent in config.entities.values():
            if ent.name == "datatype" and ent.pattern:
                import re
                match = re.search(r'\(([^)]+)\)', ent.pattern)
                if match:
                    result["datatypes"] = sorted(match.group(1).split("|"))
    except Exception:
        result["datatypes"] = []

    return result


def check_new_transformations():
    """Check for new transformation types."""
    transforms = []
    try:
        from bids.modeling.transformations import compute, munge
        for mod in [compute, munge]:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if not name.startswith("_") and hasattr(obj, "_transform"):
                    transforms.append(name)
    except Exception:
        pass
    return sorted(transforms)


def main():
    print("=" * 60)
    print("  PyBIDS Change Detection")
    print("=" * 60)

    # Current API
    current_api = extract_public_api()
    current_meta = extract_entities_and_datatypes()
    current_transforms = check_new_transformations()

    current_state = {
        "api": current_api,
        "entities": current_meta.get("entities", []),
        "datatypes": current_meta.get("datatypes", []),
        "transforms": current_transforms,
    }

    # Load previous known state
    if KNOWN_FILE.exists():
        with open(KNOWN_FILE) as f:
            known = json.load(f)
    else:
        known = {"api": {}, "entities": [], "datatypes": [], "transforms": []}

    # ── Compare modules ──
    new_modules = set(current_api.keys()) - set(known.get("api", {}).keys())
    if new_modules:
        alerts.append(f"🆕 NEW MODULES: {sorted(new_modules)}")
        for mod in sorted(new_modules):
            members = current_api[mod]
            funcs = [n for n, v in members.items() if v["type"] == "function"]
            classes = [n for n, v in members.items() if v["type"] == "class"]
            if funcs:
                alerts.append(f"   Functions: {funcs}")
            if classes:
                alerts.append(f"   Classes: {classes}")

    # ── Compare functions/classes within known modules ──
    for mod_name in sorted(set(current_api.keys()) & set(known.get("api", {}).keys())):
        current_members = set(current_api[mod_name].keys())
        known_members = set(known["api"][mod_name].keys())
        new_members = current_members - known_members
        if new_members:
            alerts.append(f"🆕 NEW in {mod_name}: {sorted(new_members)}")

        # Check for new methods on known classes
        for member_name in sorted(current_members & known_members):
            cur = current_api[mod_name][member_name]
            prev = known["api"][mod_name][member_name]
            if cur["type"] == "class" and prev["type"] == "class":
                new_methods = set(cur.get("methods", [])) - set(prev.get("methods", []))
                if new_methods:
                    alerts.append(f"🆕 NEW methods on {mod_name}.{member_name}: {sorted(new_methods)}")

    # ── Compare entities ──
    new_entities = set(current_meta.get("entities", [])) - set(known.get("entities", []))
    if new_entities:
        alerts.append(f"🆕 NEW ENTITIES: {sorted(new_entities)}")

    # ── Compare datatypes ──
    new_datatypes = set(current_meta.get("datatypes", [])) - set(known.get("datatypes", []))
    if new_datatypes:
        alerts.append(f"🆕 NEW DATATYPES: {sorted(new_datatypes)}")

    # ── Compare transforms ──
    new_transforms = set(current_transforms) - set(known.get("transforms", []))
    if new_transforms:
        alerts.append(f"🆕 NEW TRANSFORMS: {sorted(new_transforms)}")

    # ── Report ──
    print(f"\nPyBIDS version: ", end="")
    try:
        import bids
        print(bids.__version__)
    except:
        print("unknown")

    print(f"Modules scanned: {len(current_api)}")
    print(f"Total classes/functions: {sum(len(v) for v in current_api.values())}")
    print(f"Entities: {len(current_meta.get('entities', []))}")
    print(f"Datatypes: {len(current_meta.get('datatypes', []))}")
    print(f"Transforms: {len(current_transforms)}")

    if alerts:
        print(f"\n⚠️  {len(alerts)} CHANGES DETECTED:")
        for a in alerts:
            print(f"  {a}")
        print("\nAction required: implement these in bids-rs")
    else:
        print("\n✅ No new changes detected — bids-rs is up to date")

    # Save current state as new known baseline
    with open(KNOWN_FILE, "w") as f:
        json.dump(current_state, f, indent=2, default=str)
    print(f"\nSaved current API snapshot to {KNOWN_FILE}")

    # Exit with code 1 if there are alerts
    if alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
