import pytest


def test_recipe_rejects_wrong_granularity_rate_and_ratio(tmp_path):
    from openwam.train.goal_memory import load_config
    import yaml
    base = dict(stage=1, representation='goal_memory', history=dict(fps=1, recent_frames=2),
                dataset=dict(subtask_probability=.7), max_updates=10, checkpoint_every=5,
                learning_rate=1e-5, new_learning_rate=1e-4)
    def load(value):
        path = tmp_path / 'recipe.yaml'; path.write_text(yaml.safe_dump(value))
        return load_config(path)
    assert load(base)['dataset']['subtask_probability'] == .7
    for change in (dict(history=dict(fps=4)), dict(dataset=dict(subtask_probability=1.1)),
                   dict(representation='visual_history_masked_sa'), dict(stage=3),
                   dict(history=dict(fps=1, recent_frames=40))):
        with pytest.raises(ValueError):
            load({**base, **change})


def test_native_training_continuation_matches_uninterrupted(tmp_path):
    import torch
    from openwam.model.goal.goal_memory import GoalMemoryExpert, GoalMemoryConfig
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.train.goal_memory import train_updates
    dit_config = dict(dim=24, in_dim=48, out_dim=48, ffn_dim=48, text_dim=16,
                      freq_dim=16, num_heads=2, num_layers=2, eps=1e-6, patch_size=(1,2,2),
                      has_image_input=False)
    def model():
        torch.manual_seed(44)
        return GoalMemoryExpert(WanModel(**dit_config), GoalMemoryConfig(
            latent_shape=(48,1,6,4), num_goal_tokens=3, history_dropout=.15))
    def sample(index):
        return dict(current_latents=torch.randn(1,48,1,6,4), memory_latents=torch.randn(1,1,48,1,6,4),
                    target_latents=torch.randn(1,48,1,6,4), text=torch.randn(1,4,16),
                    target_uv=torch.rand(1,2,2), target_image_valid=torch.ones(1,dtype=torch.bool),
                    target_point_valid=torch.ones(1,2,dtype=torch.bool), target_in_frame=torch.ones(1,2,dtype=torch.bool))
    config = dict(stage=1,max_updates=4, checkpoint_every=2, learning_rate=1e-4,
                  new_learning_rate=1e-3, dataset=dict(subtask_probability=.7))
    kwargs = dict(config=config, dependencies={'vae':'fixture'},dit_config=dit_config,population_identity='fixture')
    full=model(); train_updates(full,sample,output_dir=tmp_path/'full',**kwargs)
    first=model(); result=train_updates(first,sample,output_dir=tmp_path/'split',stop_after=2,**kwargs)
    assert result['status']=='interrupted' and result['update']==2
    resumed=model(); result=train_updates(resumed,sample,output_dir=tmp_path/'split',resume=tmp_path/'split/resume.pt',**kwargs)
    assert result['status']=='complete' and result['update']==4
    for name, value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)


def test_policy_optimizer_owns_each_parameter_once_with_native_lrs():
    from test_goal_memory_core import fixture
    from openwam.train.goal_memory_stage2 import make_smoke_policy
    from openwam.train.goal_memory import parameter_groups
    video,_=fixture();policy=make_smoke_policy(video,{})
    groups=parameter_groups(policy,learning_rate=1e-5,new_learning_rate=1e-4,projection_learning_rate=1e-3)
    rates={id(p):g['lr'] for g in groups for p in g['params']}
    all_ids=[id(p) for g in groups for p in g['params']]
    assert len(all_ids)==len(set(all_ids))==sum(p.requires_grad for p in policy.parameters())
    assert rates[id(policy.video.dit.blocks[0].self_attn.q.weight)]==1e-5
    assert rates[id(policy.video.goal_queries)]==1e-4
    assert rates[id(policy.action_backbone.action_encoder.weight)]==1e-4
    assert rates[id(policy.proprio_encoder.weight)]==1e-3


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('destination', ['new', 'stale', 'missing'])
def test_completed_resume_publishes_restored_weights_without_consuming_rng(tmp_path, stage, destination):
    import json
    import torch
    from safetensors import safe_open
    from openwam.model.goal.goal_memory import GoalMemoryExpert, GoalMemoryConfig
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.train.goal_memory import train_updates
    from openwam.train.goal_memory_stage2 import make_smoke_policy
    dit = dict(dim=24,in_dim=48,out_dim=48,ffn_dim=48,text_dim=16,freq_dim=16,
               num_heads=2,num_layers=2,eps=1e-6,patch_size=(1,2,2),has_image_input=False)
    def model():
        video = GoalMemoryExpert(WanModel(**dit), GoalMemoryConfig(num_goal_tokens=3), dit_config=dit)
        value = video if stage == 1 else make_smoke_policy(video, {})
        value.training_loss = lambda **kw: {'loss': sum(p.float().square().sum() for p in value.parameters())}
        return value
    config = dict(stage=stage,max_updates=1,checkpoint_every=1,learning_rate=1e-4,new_learning_rate=1e-3, dtype='bfloat16')
    kwargs = dict(config=config,dependencies={},dit_config=dit,population_identity='fixture')
    original = tmp_path/'original'
    train_updates(model(), lambda _: {}, output_dir=original, **kwargs)
    checkpoint = original/'resume.pt'
    payload = torch.load(checkpoint, weights_only=False)
    output = tmp_path/'new' if destination == 'new' else original
    name = 'goal_memory.safetensors' if stage == 1 else 'policy.safetensors'
    if destination == 'missing':
        (original/name).unlink()
    elif destination == 'stale':
        from safetensors.torch import save_file
        save_file({'stale': torch.ones(1)}, str(original/name))
    def no_batches(_):
        raise AssertionError('completed resume must not consume data')
    result = train_updates(model(), no_batches, output_dir=output, resume=checkpoint, **kwargs)
    assert result['update'] == 1 and result['status'] == 'complete'
    assert torch.equal(torch.get_rng_state(), payload['torch_rng'])
    with safe_open(str(output/name), framework='pt') as file:
        key = 'goal_memory_bundle' if stage == 1 else 'goal_memory_policy'
        training = json.loads(file.metadata()[key])['training']
        assert training['update'] == 1
        assert training['config']['dtype'] == 'bfloat16'
        assert training['execution'] == result['execution']
        assert {v['dtype'] for v in training['execution']['tensors'].values()} == {'torch.float32'}
        assert {v['device'] for v in training['execution']['tensors'].values()} == {'cpu'}
        assert training['execution']['autocast']['cpu']['enabled'] is False
        for k,v in payload['model'].items():
            torch.testing.assert_close(file.get_tensor(k), v, atol=0, rtol=0)
    republished = torch.load(result['resume'], weights_only=False)
    assert republished['update'] == payload['update']
    assert torch.equal(republished['torch_rng'], payload['torch_rng'])
