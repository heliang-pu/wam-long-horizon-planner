# ----------------------------------------------------------------------------
# Long-horizon meta adapter for scripts/eval_policy.py (policy_name longhorizon): the subtask chain of
# long_horizon.py around inner_policy (phoenix, internw0 or openwam). The inner adapter's own settings (<inner>_* keys,
# checkpoint_dir) come from policy/<inner>/deploy_policy.yml unless this config or --overrides sets them.
# run_real does not go through here: --long_horizon wraps the --policy adapter it already loaded.
#
# Unified eval interface:
#   get_model(usr_args) -> model
#   eval(env, model, obs) -> (obs, info, truncated, inference_times)
#   reset_model(model) -> None
# ----------------------------------------------------------------------------

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from policy.longhorizon.long_horizon import LongHorizonModel
from policy.longhorizon.long_horizon import LongHorizonPolicy

POLICY_ROOT = Path(__file__).resolve().parent
INNER_POLICIES = ("phoenix", "internw0", "openwam")


def inner_args(usr_args: dict) -> dict:
    """usr_args with the inner adapter's deploy_policy.yml filled in where they leave it unset."""
    name = str(usr_args.get("inner_policy") or "phoenix")
    if name not in INNER_POLICIES:
        raise ValueError(f"inner_policy must be one of {INNER_POLICIES}, got {name!r}")
    defaults = yaml.safe_load((POLICY_ROOT.parent / name / "deploy_policy.yml").read_text()) or {}
    args = dict(usr_args)
    for key, value in defaults.items():
        if (key.startswith(f"{name}_") or key == "checkpoint_dir") and args.get(key) in (None, ""):
            args[key] = value
    args["policy_name"] = name
    return args


def get_model(usr_args):
    args = inner_args(usr_args)
    inner = importlib.import_module(f"policy.{args['policy_name']}.deploy_policy")
    return LongHorizonPolicy(inner).get_model(args)


def eval(env, model: LongHorizonModel, obs):
    return model.eval(env, obs)


def reset_model(model: LongHorizonModel):
    model.reset()


def close_model(model: LongHorizonModel):
    model.close()
