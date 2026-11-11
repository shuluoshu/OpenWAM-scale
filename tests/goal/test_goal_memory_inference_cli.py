import json
import pytest
import torch


@pytest.mark.parametrize('joint',[False,True])
def test_portable_policy_inference_cli_without_training_assets(tmp_path,joint):
    from scripts.train_goal_memory import smoke
    from scripts.infer_goal_memory import main
    from openwam.train.goal_memory_stage2 import save_policy_bundle
    config=dict(stage=2,seed=42,memory={},policy=dict(
        attention_mode='action_sees_video' if joint else 'isolated',
        future_video_mode='joint' if joint else 'none'))
    policy,provider,_,dependencies,_=smoke(config,'cpu',torch.float32)
    batch=provider(0)
    inputs={key:batch[key] for key in ('current_latents','memory_latents','text','proprio',
                                     'memory_frame_ids','memory_ages','goal_horizon_s')}
    checkpoint=tmp_path/'policy.safetensors'
    save_policy_bundle(policy,checkpoint,dependencies=dependencies)
    del policy,provider
    torch.save(inputs,tmp_path/'inputs.pt')
    (tmp_path/'deps.json').write_text(json.dumps(dependencies))
    assert main(['--checkpoint',str(checkpoint),'--dependencies',str(tmp_path/'deps.json'),
                 '--inputs',str(tmp_path/'inputs.pt'),'--output',str(tmp_path/'actions.pt'),
                 '--device','cpu','--dtype','float32','--steps','2'])==0
    out=torch.load(tmp_path/'actions.pt',weights_only=True)
    assert out['normalized_actions'].shape==(1,32,112)
    assert torch.isfinite(out['normalized_actions']).all()
    assert out['format']=='openwam.goal_memory.actions.v1'
