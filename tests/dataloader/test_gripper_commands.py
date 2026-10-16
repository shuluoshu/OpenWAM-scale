import numpy as np
import pytest

from openwam.dataloader.utils.gripper import GripperCommands, gripper_state_outliers
from openwam.dataloader.utils.mora_relative import MoraRelativeRepresentation


def test_commands_are_independent_of_state_and_joint_next_state():
    info = {"features": {"eef": {"shape": [20]}, "joints": {"shape": [2], "names": ["left_joint", "right_joint"]},
                         "cmd": {"shape": [2]}}}
    rep = MoraRelativeRepresentation(info, state_key="eef", joint_state_key="joints",
                                    gripper_command_sources={"left": {"key": "cmd", "index": 0},
                                                             "right": {"key": "cmd", "index": 1}})
    eef = np.zeros((4, 20), np.float32)
    eef[:, [9, 19]] = [[.2, .3], [.4, .5], [.6, .7], [.8, .9]]
    cmd = np.array([[1.4, -.2], [.1, .2], [.7, .8], [.9, 1.]])
    action, mask, state, sm = rep.encode_window({"eef": eef, "joints": [[1, 10], [2, 12], [4, 15], [8, 19]], "cmd": cmd})
    np.testing.assert_allclose(state[0, [9, 19]], [.2, .3])
    np.testing.assert_allclose(action[:, [9, 19]], [[1, 0], [.1, .2], [.7, .8]])
    np.testing.assert_allclose(action[:, 20], [1, 3, 7])
    np.testing.assert_allclose(action[:, 27], [2, 5, 9])
    assert mask[:, [9, 19]].all() and sm[:, [9, 19]].all()


def test_missing_command_masks_gt_without_losing_measured_state():
    info = {"features": {"eef": {"shape": [20]}, "action": {"shape": [20]},
                         "actions.source_action": {"shape": [20]}}}
    rep = MoraRelativeRepresentation(info, state_key="eef", joint_state_key="absent")
    eef = np.ones((3, 20))
    action, mask, state, sm = rep.encode_window({"eef": eef})
    assert not mask[:, [9, 19]].any()
    assert not action[:, [9, 19]].any()
    assert sm[:, [9, 19]].all() and (state[:, [9, 19]] == 1).all()


def test_scalar_shifted_command_alignment_masks_boundary_and_nonfinite():
    g = GripperCommands({"features": {"cmd": {"shape": []}}}, {"left": {"key": "cmd", "row_offset": -1}})
    values, valid = g.align({"cmd": [0.2, np.nan, .9, .1]}, 4)
    np.testing.assert_allclose(values[:, 0], [0, .2, 0, .9])
    np.testing.assert_array_equal(valid[:, 0], [False, True, False, True])
    assert not valid[:, 1].any()


def test_auto_uses_master_openness_but_never_guess_source_action():
    info = {"features": {"master_actions.left_gripper.openness": {"shape": [1]},
                         "actions.source_action": {"shape": [14]}}}
    g = GripperCommands(info)
    assert g.sources["left"]["key"] == "master_actions.left_gripper.openness"
    assert "right" not in g.sources
    with pytest.raises(ValueError, match="absent"):
        GripperCommands(info, {"left": {"key": "misspelled"}})


def test_auto_restores_documented_shifted_master_command():
    info = {"features": {"actions.source_action": {"shape": [2], "names": ["left_gripper", "right_gripper"],
                                                     "gripper_unit": "openness_0_closed_1_open"}},
            "starvla_metadata": {"action": {"source_action_field": "actions.source_action",
                                          "source_action_semantic": {"frame_offset": 1, "source": "source_action_trajectory_shifted"}},
                                 "eef_action_source_keys": ["raw.master.joint_position_left", "raw.master.joint_position_right"]}}
    g = GripperCommands(info)
    assert all(s["row_offset"] == -1 for s in g.sources.values())


def test_filter_rejects_stable_command_spikes_but_keeps_real_open_close():
    state = np.full((70, 2), .5)
    commands = np.full((70, 2), .5);valid = np.ones((70, 2), bool)
    state[30:33, 1] = 1
    bad, reasons = gripper_state_outliers(state, commands, valid, {})
    assert bad[30:33].all() and not bad[:28].any()
    assert reasons["stable_command_jump"] > 0
    # Actual rapid commanded opening and closing should survive.
    commands[30:33, 1] = 1
    bad, _ = gripper_state_outliers(state, commands, valid, {})
    assert not bad.any()


def test_filter_detects_short_return_without_commands_and_respects_disabled():
    state = np.full((20, 2), .25);state[8, 0] = .9
    commands = np.zeros_like(state);valid = np.zeros_like(state, bool)
    bad, _ = gripper_state_outliers(state, commands, valid, {})
    assert bad[8] and not bad[:7].any()
    bad, _ = gripper_state_outliers(state, commands, valid, {"enabled": False})
    assert not bad.any()
    # Sustained monotonic open transition is not an isolated spike.
    state[8:, 0] = .9
    bad, _ = gripper_state_outliers(state, commands, valid, {})
    assert not bad.any()


def test_filter_does_not_treat_missing_labels_as_spikes():
    state = np.full((40, 2), np.nan)
    bad, _ = gripper_state_outliers(state, np.zeros_like(state), np.zeros_like(state, bool), {})
    assert not bad.any()


def test_state_derived_source_action_is_never_promoted_to_cmd():
    info = {"features": {"actions.source_action": {"shape": [2], "names": ["left_gripper", "right_gripper"],
                                                     "gripper_unit": "openness_0_closed_1_open"}},
            "starvla_metadata": {"action": {"source_action_field": "actions.source_action",
                                            "source_action_semantic": {"frame_offset": 1,
                                                "source": "Stage2 action next_step(observation.state)"}}}}
    assert not GripperCommands(info).sources


def test_declared_raw_master_scale_and_unknown_filter_option():
    key = "raw.master.end_effector_left_position_align"
    info = {"features": {key: {"shape": []}}, "starvla_metadata": {"gripper_normalization": {
        "normalization_rules": {"left": {"action_source_field": key, "source_convention": "0=closed, 10=open",
                                         "source_clip_range": [0, 10]}}}}}
    commands = GripperCommands(info)
    values, valid = commands.align({key: [-2, 5, 14]}, 3)
    np.testing.assert_allclose(values[:, 0], [0, .5, 1])
    assert valid[:, 0].all() and not valid[:, 1].any()
    with pytest.raises(ValueError, match="unknown"):
        gripper_state_outliers(values, values, valid, {"max_stepp": .5})


def human_info(source):
    formulas = {"LightWheel": "min_thumb_tip_to_fingertips_divided_by_0.1_m",
                "mecka": "min(thumb_tip->fingertips)/0.1513m_clipped_0_1",
                "egodex": "clipped_distance_after_closedness_flip",
                "egoverse": "clipped_distance_after_closedness_flip"}
    return {"features": {"eef": {"shape": [20]}, "actions.eef": {"shape": [20]}},
            "starvla_metadata": {"source": source, "embodiment": "human_hands",
                "gripper_normalization": {"status": "normalized", "range": [0, 1],
                    "convention": "0=closed, 1=open", "normalization": formulas[source]}}}


@pytest.mark.parametrize("source", ["LightWheel", "mecka", "egodex", "egoverse"])
def test_audited_human_targets_supervise_gripper_without_tcp(source):
    info = human_info(source)
    rep = MoraRelativeRepresentation(info, state_key="eef", joint_state_key="absent")
    eef = np.zeros((4, 20), np.float32)
    eef[:, [9, 19]] = [[.1, .2], [.2, .3], [.3, .4], [.4, .5]]
    target = np.zeros_like(eef)
    target[:, [9, 19]] = [[.6, .7], [.7, .8], [.8, .9], [.9, 1]]
    action, mask, state, sm = rep.encode_window({"eef": eef, "actions.eef": target})
    assert mask[:, [9, 19]].all()
    np.testing.assert_array_equal(action[:, [9, 19]], target[:-1, [9, 19]])
    np.testing.assert_array_equal(state[0, [9, 19]], eef[0, [9, 19]])
    assert sm[:, [9, 19]].all() and mask.sum() == 6
    assert all(s["kind"] == "human_openness_target" and s["row_offset"] == 0
               for s in rep.gripper_commands.sources.values())
    assert rep.contract["gripper_target_source"] == "controller_command_or_human_openness_target"


def test_human_target_mask_honors_nonfinite_and_declared_validity():
    info = human_info("egodex")
    info["features"]["action.eef_target_valid"] = {"shape": []}
    g = GripperCommands(info)
    target = np.zeros((4, 20), np.float32)
    target[:, [9, 19]] = [[.4, .5], [.6, .7], [np.nan, .8], [1.2, -.2]]
    values, valid = g.align({"actions.eef": target, "action.eef_target_valid": [True, False, True, True]}, 4)
    np.testing.assert_array_equal(valid, [[True, True], [False, False], [False, True], [True, True]])
    np.testing.assert_allclose(values, [[.4, .5], [0, 0], [0, .8], [1, 0]])


def test_real_commands_take_priority_over_human_targets_per_side():
    info = human_info("LightWheel")
    key = "master_actions.left_gripper.openness"
    info["features"][key] = {"shape": []}
    g = GripperCommands(info)
    assert g.sources["left"]["key"] == key
    assert g.sources["right"]["key"] == "actions.eef"
    valid = np.ones((3, 2), bool)
    np.testing.assert_array_equal(g.controller_valid(valid), [[True, False]] * 3)
    assert valid.all()  # Filtering must not mutate supervision validity.


@pytest.mark.parametrize("change", ["robot", "unknown_source", "unknown_formula", "wrong_range", "disabled", "missing_target"])
def test_human_recognition_requires_audited_schema(change):
    info = human_info("LightWheel")
    meta = info["starvla_metadata"]
    if change == "robot": meta["embodiment"] = "robot_bimanual"
    if change == "unknown_source": meta["source"] = "unknown_human_capture"
    if change == "unknown_formula": meta["gripper_normalization"]["normalization"] = "unknown"
    if change == "wrong_range": meta["gripper_normalization"]["range"] = [-1, 1]
    if change == "missing_target": del info["features"]["actions.eef"]
    assert not GripperCommands(info, {} if change == "disabled" else None).sources


def test_human_proxy_echo_is_not_evidence_to_accept_state_spike():
    g = GripperCommands(human_info("LightWheel"))
    state = np.full((40, 2), .3, np.float32)
    state[20, 0] = .9
    target = np.zeros((40, 20), np.float32)
    target[:-1, [9, 19]] = state[1:]
    target[-1, [9, 19]] = state[-1]
    values, valid = g.align({"actions.eef": target}, len(state))
    assert valid.all()
    bad, _ = gripper_state_outliers(state, values, g.controller_valid(valid))
    assert bad[20]
