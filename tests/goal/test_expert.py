"""Real small Wan blocks exercise goal training and sampling, not a surrogate net."""
import torch
import pytest
from torch import nn
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.goal.types import GoalTargets

torch.set_num_threads(2)

TINY_DIT = dict(dim=48,in_dim=48,ffn_dim=64,out_dim=48,text_dim=16,freq_dim=16,
                eps=1e-6,patch_size=(1,2,2),num_heads=2,num_layers=1,
                has_image_input=False,seperated_timestep=True,require_vae_embedding=False,
                require_clip_embedding=False,fuse_vae_embedding_in_latents=True)

class TinyVAE(nn.Module):
    """Only the frozen IO dependency is a deterministic stand-in."""
    def __init__(self):
        super().__init__(); self.calls=[]; self.register_buffer('marker',torch.zeros(()))
    def encode(self, videos, device, **kwargs):
        x=torch.stack(list(videos)).to(device); self.calls.append(tuple(x.shape))
        assert x.shape[2]==1, 'discontinuous frames must never enter temporal VAE together'
        return torch.nn.functional.avg_pool3d(x,(1,16,16)).repeat(1,16,1,1,1)
    def decode(self, z, device, **kwargs):
        return torch.nn.functional.interpolate(z[:,:3].to(device),size=(1,256,320))


def make_expert(**kwargs):
    from openwam.model.goal.codec import GoalCodec
    from openwam.model.goal.expert import GoalExpert
    torch.manual_seed(5)
    return GoalExpert(WanModel(**TINY_DIT), GoalCodec(TinyVAE()),dit_config=TINY_DIT,**kwargs)


def batch(n=2):
    torch.manual_seed(3)
    image=torch.rand(n,3,1,256,320)*2-1
    targets=GoalTargets(image*.5,torch.ones(n,dtype=torch.bool),
                        torch.tensor([[[80/319,90/255],[240/319,160/255]]]).repeat(n,1,1),
                        torch.ones(n,2,dtype=torch.bool),torch.ones(n,2,dtype=torch.bool),
                        torch.tensor([[-1,0]],dtype=torch.int8).repeat(n,1))
    return image,torch.randn(n,4,16),targets


def test_independent_single_frame_codec_and_actual_wan_gradient():
    e=make_expert(); obs,ctx,t=batch()
    result=e(obs,ctx,t,timestep_indices=torch.tensor([200,700]))
    assert e.codec.vae.calls==[(2,3,1,256,320)]*2
    assert result['clean_latent'].shape==(2,48,1,16,20)
    result['loss'].backward()
    assert e.dit.patch_embedding.weight.grad.abs().sum()>0
    assert not any('vae' in key for key in e.state_dict())


def test_point_loss_reaches_goal_generator_and_masks_unknown_occluded_independently():
    e=make_expert();obs,ctx,t=batch()
    out=e(obs,ctx,t,timestep_indices=torch.tensor([600,800]));out['kp_loss'].backward()
    assert out['valid_points']==4 # unknown and occluded geometry are trainable
    assert e.dit.head.head.weight.grad.abs().sum()>0
    assert e.point_head.net[-1].weight.grad.abs().sum()>0


def test_empty_nan_point_mask_is_finite_zero_with_zero_gradient():
    e=make_expert();obs,ctx,t=batch();t.point_valid[:]=False;t.uv[:]=float('nan')
    out=e(obs,ctx,t);assert out['valid_points']==0
    assert torch.isfinite(out['kp_loss']) and out['kp_loss'].item()==0
    out['kp_loss'].backward()
    assert e.dit.head.head.weight.grad.abs().sum()==0


def test_goal_only_fm_uses_scheduler_sigmas_and_batch_training_weights():
    e=make_expert();obs,ctx,t=batch();idx=torch.tensor([100,800]);noise=torch.randn(2,48,1,16,20)
    out=e(obs,ctx,t,timestep_indices=idx,noise=noise)
    clean=e.codec.encode(t.image);sigma=e.scheduler.sigmas[idx].reshape(2,1,1,1,1)
    assert torch.equal(out['sigma'],sigma.flatten())
    assert torch.allclose(out['noisy_latent'],(1-sigma)*clean+sigma*noise)
    assert torch.allclose(out['clean_latent'],out['noisy_latent']-sigma*out['velocity'])
    mse=(out['velocity'].float()-(noise-clean).float()).square().flatten(1).mean(1)
    expected=(mse*e.scheduler.linear_timesteps_weights[idx]).mean()
    assert torch.allclose(out['fm_loss'],expected)
    assert torch.equal(out['observed_latent'],e.codec.encode(obs))


def test_mask_safe_weighted_heatmaps_left_right_order_and_finite_uv():
    from openwam.model.goal.heads import heatmap_loss,heatmap_uv
    logits=torch.full((2,2,64,80),-30.);logits[:,0,20,10]=30;logits[:,1,40,60]=30
    uv=heatmap_uv(logits)
    assert (uv[:,0,0]<uv[:,1,0]).all()
    _,_,t=batch();t.uv[0,0]=float('nan');t.image_valid[1]=False
    loss,count=heatmap_loss(logits.requires_grad_(),t,torch.tensor([.5,.9]))
    assert count==1 and torch.isfinite(loss)
    loss.backward();assert logits.grad[1].abs().sum()==0


def test_observation_tokens_cannot_read_noisy_goal():
    e=make_expert();obs,ctx,_=batch(1);z=e.codec.encode(obs)
    a=e.velocity(z,torch.zeros_like(z),torch.tensor([700.]),ctx,return_observation=True)
    b=e.velocity(z,torch.ones_like(z),torch.tensor([700.]),ctx,return_observation=True)
    assert torch.allclose(a[1],b[1],atol=1e-6)
    assert not torch.allclose(a[0],b[0])


def test_generate_no_gt_deterministic_final_latent_and_scheduler_unchanged():
    e=make_expert().eval();obs,ctx,_=batch(1);train_sigmas=e.scheduler.sigmas.clone()
    a=e.generate(obs,ctx,seed=91,num_steps=3);b=e.generate(obs,ctx,seed=91,num_steps=3)
    assert a.latent.shape==(1,48,1,16,20) and torch.equal(a.latent,b.latent)
    assert a.point_logits.shape==(1,2,64,80) and a.token_valid.shape==(1,80)
    assert torch.equal(train_sigmas,e.scheduler.sigmas)
    assert not a.latent.requires_grad


def test_bad_codec_shape_and_text_dimensions_fail_explicitly():
    e=make_expert();obs,ctx,t=batch(1)
    with pytest.raises(ValueError,match='256'):e.codec.encode(obs[:,:,:,:128])
    with pytest.raises(ValueError,match='text'):e(obs,ctx[:,:,:8],t)
