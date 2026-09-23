from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def test_smoke_training_config_composes_to_two_bounded_sources():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="openwam_alpha_v0_1_smoke")

    assert cfg.model.architecture.action_dim == 80
    assert cfg.model.architecture.state_dim == 80
    assert cfg.model.architecture.framework == "dual_system"
    assert cfg.training.finetune_ckpt_path.endswith("OpenWAM-Alpha-Pretrain-Foundation-Model")
    assert cfg.training.batch_size == 1
    assert cfg.training.max_steps == 2
    assert cfg.training.num_epochs is None
    assert cfg.training.gradient_accumulation_steps == 1
    assert cfg.training.mixed_precision == "bf16"
    assert cfg.training.zero_stage == 2
    assert cfg.training.use_gradient_checkpointing is True
    assert cfg.training.offload_optimizer_device == "cpu"
    assert cfg.training.dataset_num_workers == 0
    assert cfg.training.save_steps == 0
    assert cfg.training.lambda_video == 1.0
    assert cfg.training.lambda_action == 1.0
    assert cfg.project.seed == 42
    assert cfg.dataloader.type == "mixture"
    assert list(cfg.dataloader.datasets) == ["interndata_genie1", "agibotworld_a2d"]
    for source in cfg.dataloader.datasets.values():
        assert source.type == "canonical_eef_lerobot"
        assert source.episode_indices == [0, 1]
        assert source.max_episodes == 2
        assert source.max_windows == 8
        assert source.num_frames == 33
        assert source.video_stride == 4
        assert source.unify_action_map == ["0-9", "34-43"]
        assert source.return_trace is False
    assert not OmegaConf.missing_keys(cfg)


def test_four_profiles_use_exact_confirmed_paths_and_contracts():
    expected = {
        "canonical_eef_interndata_smoke.yaml": "pretrain_datasetsV0_1/interndata_a1/",
        "canonical_eef_agibotworld_smoke.yaml": "pretrain_datasetsV0_1/agibotworld/",
        "canonical_eef_robodojo_openloop.yaml": "post_train_data/robodojo_sim/",
        "canonical_eef_robotwin_openloop.yaml": "post_train_data/robotwin_traj_fullfield/",
    }
    for filename, path_fragment in expected.items():
        cfg = OmegaConf.load(CONFIG_DIR / "dataloader" / filename)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert resolved["type"] == "canonical_eef_lerobot"
        assert path_fragment in resolved["dataset_dir"]
        assert resolved["num_frames"] == 33
        assert resolved["video_stride"] == 4
        assert resolved["height"] == 384
        assert resolved["width"] == 320
        assert resolved["multiview"] is True
        assert resolved["normalize_mode"] == "quantile"
        assert resolved["unify_action"] is True
        assert resolved["unify_action_map"] == ["0-9", "34-43"]
        assert resolved["episode_indices"] == [0, 1]
        assert resolved["max_episodes"] == 2
        assert resolved["max_windows"] == 8
        if "openloop" in filename:
            assert resolved["return_trace"] is True
