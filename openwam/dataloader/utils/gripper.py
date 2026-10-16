"""Measured inputs, controller/human gripper targets, and spike rejection."""
from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
ALIGNED_COMMAND = "_aligned_gripper_command"
ALIGNED_VALID = "_aligned_gripper_command_valid"
HUMAN_TARGET = "human_openness_target"
HUMAN_NORMALIZATIONS = {
    "LightWheel": "min_thumb_tip_to_fingertips_divided_by_0.1_m",
    "mecka": "min(thumb_tip->fingertips)/0.1513m_clipped_0_1",
    "egodex": "clipped_distance_after_closedness_flip",
    "egoverse": "clipped_distance_after_closedness_flip",
}
FILTER_DEFAULTS = dict(enabled=True, max_step=0.1, command_tolerance=0.01,
                       settling_frames=15, reversal_window=3, return_tolerance=0.02)


def filter_config(config=None):
    config = dict(config or {})
    if set(config) - set(FILTER_DEFAULTS):
        raise ValueError(f"unknown gripper_state_filter keys: {set(config) - set(FILTER_DEFAULTS)}")
    result = {**FILTER_DEFAULTS, **config}
    if not isinstance(result["enabled"], bool):
        raise ValueError("gripper_state_filter.enabled must be boolean")
    for key in ("max_step", "command_tolerance", "return_tolerance"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"gripper_state_filter.{key} must be finite and positive")
    for key in ("settling_frames", "reversal_window"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"gripper_state_filter.{key} must be a positive integer")
    return result


def _names(feature):
    names = feature.get("names") or []
    return names.get("motors", []) if isinstance(names, dict) else names


def _auto_sources(info):
    """Recognize audited schemas only; 'action' alone is not command evidence."""
    features = info.get("features", {})
    meta = info.get("starvla_metadata", {})
    sources = {}
    for side in SIDES:
        key = f"master_actions.{side}_gripper.openness"
        if key in features:
            sources[side] = dict(key=key, provenance="recorded master gripper openness")
        rule = meta.get("gripper_normalization", {}).get("normalization_rules", {}).get(side, {})
        key = rule.get("action_source_field", "")
        if (side not in sources and key.startswith("raw.master.") and key in features
                and rule.get("source_convention") == "0=closed, 10=open"
                and rule.get("source_clip_range") == [0.0, 10.0]):
            sources[side] = dict(key=key, scale=0.1, provenance="declared raw.master command; 0..10 to 0..1")
    action_meta = meta.get("action", {})
    semantic = action_meta.get("source_action_semantic", {})
    key = action_meta.get("source_action_field")
    known_same_tick = semantic.get("semantic") == "absolute_same_frame_controller_command" and semantic.get("frame_offset") == 0
    known_shifted_master = (
        semantic.get("source") == "source_action_trajectory_shifted"
        and type(semantic.get("frame_offset")) is int
        and any(str(k).startswith("raw.master.") for k in meta.get("eef_action_source_keys", [])))
    if (key in features and (known_same_tick or known_shifted_master)
            and features[key].get("gripper_unit") == "openness_0_closed_1_open"):
        names = _names(features[key])
        for side in SIDES:
            if side not in sources and f"{side}_gripper" in names:
                sources[side] = dict(key=key, index=names.index(f"{side}_gripper"),
                                     row_offset=-int(semantic["frame_offset"]),
                                     provenance="source_action_semantic: " + str(semantic))
    key = "action.position"
    if (features.get(key, {}).get("status") == "diagnostic_preserved_from_stage2"
            and features[key].get("source_key") == key
            and meta.get("gripper_normalization", {}).get("ontology_version") == "morphirobotV1_29d"
            and meta["gripper_normalization"].get("source_convention")
            == "observation.state and action gripper values are source openness"):
        names = _names(features[key])
        for side, name in zip(SIDES, ("Lgripper_Joint", "Rgripper_Joint")):
            if side not in sources and name in names:
                sources[side] = dict(key=key, index=names.index(name),
                                     provenance="preserved Morphi stage2 action.position openness")
    # Human demonstrations have geometric openness GT rather than a robot cmd.
    # Preserve the stored target's timing: actions.eef already contains the
    # future value, and generic action can have a different offset (LightWheel).
    norm = meta.get("gripper_normalization", {})
    source = meta.get("source")
    if (source in HUMAN_NORMALIZATIONS
            and meta.get("embodiment") in {"human_hands", "human_bimanual"}
            and norm.get("status") == "normalized" and norm.get("range") == [0, 1]
            and norm.get("convention") == "0=closed, 1=open"
            and norm.get("normalization") == HUMAN_NORMALIZATIONS[source]
            and features.get("actions.eef", {}).get("shape") == [20]):
        validity = ({"valid_key": "action.eef_target_valid"}
                    if "action.eef_target_valid" in features else {})
        for side, index in zip(SIDES, (9, 19)):
            sources.setdefault(side, dict(key="actions.eef", index=index, kind=HUMAN_TARGET,
                provenance=f"{source}: stored human openness target; {norm['normalization']}",
                **validity))
    return sources


class GripperCommands:
    """Map row i to a controller command or an audited human openness target.

    Explicit sources use {left/right: {key, index?, row_offset?, scale?, bias?}}.
    An empty mapping disables command supervision; None selects audited schemas.
    Optional kind distinguishes human targets from independent commands for
    spike filtering. valid_key/valid_index select a stored target-valid flag.
    Missing/nonfinite targets are masked, never replaced with measured state.
    """

    def __init__(self, info, sources=None):
        features = info.get("features", {})
        sources = _auto_sources(info) if sources is None else dict(sources)
        if set(sources) - set(SIDES):
            raise ValueError("gripper_command_sources may contain only left/right")
        self.sources = {}
        for side, spec in sources.items():
            spec = dict(spec)
            if set(spec) - {"key", "index", "row_offset", "scale", "bias", "provenance",
                            "kind", "valid_key", "valid_index"}:
                raise ValueError(f"unknown gripper command mapping: {side}: {spec}")
            key = spec.get("key")
            if key not in features:
                raise ValueError(f"gripper command column {key!r} absent")
            shape = features[key].get("shape")
            index = spec.get("index")
            if shape not in ([], [1]) and (not shape or len(shape) != 1 or
                    type(index) is not int or not 0 <= index < shape[0]):
                raise ValueError(f"gripper command {key}: explicit valid vector index required")
            if shape in ([], [1]) and index not in (None, 0):
                raise ValueError(f"gripper command {key}: scalar index must be 0")
            offset = spec.get("row_offset", 0)
            if type(offset) is not int:
                raise ValueError("gripper command row_offset must be an integer")
            scale, bias = float(spec.get("scale", 1)), float(spec.get("bias", 0))
            if not np.isfinite([scale, bias]).all() or scale == 0:
                raise ValueError("gripper command scale/bias must be finite, scale nonzero")
            self.sources[side] = dict(key=key, index=0 if index is None else index,
                                     row_offset=offset, scale=scale, bias=bias,
                                     provenance=spec.get("provenance", "explicit profile mapping"))
            kind = spec.get("kind", "controller_command")
            if kind not in ("controller_command", HUMAN_TARGET):
                raise ValueError(f"unknown gripper target kind: {kind!r}")
            # Keep legacy controller contracts byte-compatible with old stats.
            if kind == HUMAN_TARGET:
                self.sources[side]["kind"] = kind
            if "valid_index" in spec and "valid_key" not in spec:
                raise ValueError("gripper valid_index requires valid_key")
            if "valid_key" in spec:
                valid_key, valid_index = spec["valid_key"], spec.get("valid_index", 0)
                shape = features.get(valid_key, {}).get("shape")
                width = 1 if shape == [] else shape[0] if shape and len(shape) == 1 else 0
                if type(valid_index) is not int or not 0 <= valid_index < width:
                    raise ValueError(f"gripper validity {valid_key!r}: absent or invalid index")
                self.sources[side].update(valid_key=valid_key, valid_index=valid_index)
        self.columns = list(dict.fromkeys(key for s in self.sources.values()
                                         for key in [s["key"], *([s["valid_key"]] if "valid_key" in s else [])]))

    def align(self, columns, n):
        values, valid = np.zeros((n, 2), np.float32), np.zeros((n, 2), bool)
        for side_id, side in enumerate(SIDES):
            if side not in self.sources:
                continue
            spec = self.sources[side]
            source = np.asarray(columns[spec["key"]], dtype=np.float32).reshape(n, -1)[:, spec["index"]]
            rows = np.arange(n) + spec["row_offset"]
            keep = (rows >= 0) & (rows < n)
            dest = np.flatnonzero(keep)
            scaled = source[rows[keep]] * spec["scale"] + spec["bias"]
            good = np.isfinite(scaled)
            if "valid_key" in spec:
                flags = np.asarray(columns[spec["valid_key"]], dtype=np.float32).reshape(n, -1)[:, spec["valid_index"]]
                good &= flags[rows[keep]] == 1
            values[dest[good], side_id] = np.clip(scaled[good], 0, 1)
            valid[dest[good], side_id] = True
        return values, valid

    def controller_valid(self, valid):
        """Only independent robot commands can explain measured-state spikes."""
        result = np.asarray(valid, dtype=bool).copy()
        for i, side in enumerate(SIDES):
            if self.sources.get(side, {}).get("kind") == HUMAN_TARGET:
                result[:, i] = False
        return result


def gripper_state_outliers(state, commands, command_valid, config=None):
    """Flag normalized measured-state jumps inconsistent with command history.

    Reject large jumps after a settled constant command, or brief jump/return
    spikes without a nearby command change. A sustained uncommanded step alone
    is insufficient without command evidence. NaNs remain label-mask concerns.
    Returns per-row flags; readers reject every window intersecting these rows.
    """
    cfg = filter_config(config)
    state = np.asarray(state, dtype=np.float32)
    commands, command_valid = np.asarray(commands), np.asarray(command_valid, bool)
    if state.ndim != 2 or state.shape[1] != 2 or commands.shape != state.shape or command_valid.shape != state.shape:
        raise ValueError("gripper filter expects three aligned [frames,2] arrays")
    bad = np.zeros(len(state), bool)
    reasons = {"stable_command_jump": 0, "short_return_spike": 0}
    if not cfg["enabled"]:
        return bad, reasons
    for side in range(2):
        s, c, cv = state[:, side], commands[:, side], command_valid[:, side]
        jumps = np.flatnonzero(np.isfinite(s[1:]) & np.isfinite(s[:-1])
                               & (np.abs(np.diff(s)) > cfg["max_step"])) + 1
        for i in jumps:
            lo = max(0, i - cfg["settling_frames"])
            hi = min(len(s), i + 2)
            stable = (i >= cfg["settling_frames"] and cv[lo:hi].all()
                      and np.ptp(c[lo:hi]) <= cfg["command_tolerance"])
            if stable:
                bad[i-1:i+1] = True
                reasons["stable_command_jump"] += 1
            for end in range(i + 1, min(len(s), i + cfg["reversal_window"] + 1)):
                if not np.isfinite(s[i:end+1]).all() or abs(s[end] - s[i-1]) > cfg["return_tolerance"]:
                    continue
                # An actual commanded open/close (including actuator lag) is allowed.
                cs, vs = c[lo:end+1], cv[lo:end+1]
                changed = vs.any() and np.ptp(cs[vs]) > cfg["command_tolerance"]
                if not changed:
                    bad[i:end] = True
                    reasons["short_return_spike"] += 1
                break
    return bad, reasons
