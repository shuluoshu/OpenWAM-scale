"""Independent Wan goal DiT. No downstream video/action branch is instantiated."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from openwam.model.video_backbone.wan.models.dit import sinusoidal_embedding_1d
from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
from .heads import GoalPointHead,heatmap_loss
from .types import GoalPrediction


class GoalExpert(nn.Module):
    def __init__(self, dit, codec, *, dit_config, shift=5., lambda_kp=.1,
                 point_hidden_dim=64, heatmap_size=(64,80), use_gradient_checkpointing=False):
        super().__init__()
        if (dit.in_dim!=48 or dit.head.head.out_features!=192 or tuple(dit.patch_size)!=(1,2,2)
                or dit.has_image_input or dit.require_clip_embedding or dit.require_vae_embedding
                or not dit.fuse_vae_embedding_in_latents):
            raise ValueError('goal expert requires Wan2.2 TI2V 48-channel patch(1,2,2) configuration')
        if shift<=0 or lambda_kp<0:raise ValueError('invalid goal shift/lambda_kp')
        self.dit=dit;self.codec=codec;self.dit_config=dict(dit_config)
        self.shift=float(shift);self.lambda_kp=float(lambda_kp)
        self.use_gradient_checkpointing=use_gradient_checkpointing
        self.point_head=GoalPointHead(point_hidden_dim,heatmap_size)
        # Learned role tags are independent of time/rotary coordinates.
        self.role_embedding=nn.Parameter(torch.zeros(2,dit.dim))
        nn.init.normal_(self.role_embedding,std=.02)
        self.scheduler=FlowMatchScheduler(template='Wan')
        self.scheduler.set_timesteps(1000,training=True,shift=self.shift)

    def velocity(self, observed_latent, noisy_goal, timestep, context, *, context_lens=None,return_observation=False):
        self.codec.validate_latent(observed_latent);self.codec.validate_latent(noisy_goal)
        if context.ndim!=3 or context.shape[0]!=observed_latent.shape[0] or context.shape[-1]!=self.dit.text_embedding[0].in_features:
            raise ValueError('incompatible goal text context [B,L,text_dim]')
        d=self.dit;b=observed_latent.shape[0]
        z=torch.cat((observed_latent,noisy_goal),dim=2)
        patches=d.patchify(z);_,_,f,h,w=patches.shape;n=h*w
        tokens=patches.flatten(2).transpose(1,2)
        tokens=tokens+self.role_embedding.repeat_interleave(n,dim=0)[None]
        token_t=torch.cat((torch.zeros(b,n,device=z.device,dtype=timestep.dtype),timestep[:,None].expand(-1,n)),dim=1)
        time=d.time_embedding(sinusoidal_embedding_1d(d.freq_dim,token_t.flatten()).to(tokens.dtype)).reshape(b,2*n,d.dim)
        modulation=d.time_projection(time).unflatten(2,(6,d.dim))
        frequencies=torch.cat((d.freqs[0][:f,None,None,:].expand(f,h,w,-1),
                               d.freqs[1][None,:h,None,:].expand(f,h,w,-1),
                               d.freqs[2][None,None,:w,:].expand(f,h,w,-1)),dim=-1).reshape(2*n,1,-1).to(z.device)
        mask=torch.ones(2*n,2*n,device=z.device,dtype=torch.bool);mask[:n,n:]=False
        ctx=d.text_embedding(context)
        ctx_mask=None
        if context_lens is not None:
            context_lens=torch.as_tensor(context_lens,device=z.device)
            if context_lens.shape!=(b,) or (context_lens<=0).any() or (context_lens>context.shape[1]).any():
                raise ValueError('invalid goal context lengths')
            ctx_mask=(torch.arange(context.shape[1],device=z.device)[None,:]<context_lens[:,None])[:,None,None,:]
        for block in d.blocks:
            if self.training and self.use_gradient_checkpointing:
                tokens=checkpoint(block,tokens,ctx,modulation,frequencies,ctx_mask,mask,use_reentrant=False)
            else:tokens=block(tokens,ctx,modulation,frequencies,context_mask=ctx_mask,self_attn_mask=mask)
        output=d.unpatchify(d.head(tokens,time),(f,h,w))
        return (output[:,:,1:],output[:,:,:1]) if return_observation else output[:,:,1:]

    def forward(self, observed, context, targets, *, timestep_indices=None, noise=None,context_lens=None):
        observed_latent=self.codec.encode(observed)
        safe_image=torch.where(targets.image_valid[:,None,None,None,None],targets.image,torch.zeros_like(targets.image))
        clean=self.codec.encode(safe_image);batch=clean.shape[0]
        idx=(torch.randint(0,len(self.scheduler.timesteps),(batch,),device=clean.device)
             if timestep_indices is None else timestep_indices.to(device=clean.device,dtype=torch.long))
        if idx.shape!=(batch,) or (idx<0).any() or (idx>=1000).any():raise ValueError('invalid goal timestep indices')
        sigma=self.scheduler.sigmas.to(clean.device)[idx].to(clean.dtype)
        timesteps=self.scheduler.timesteps.to(clean.device)[idx].to(clean.dtype)
        noise=torch.randn_like(clean) if noise is None else noise
        if noise.shape!=clean.shape:raise ValueError('goal noise shape mismatch')
        s=sigma[:,None,None,None,None];noisy=(1-s)*clean+s*noise
        velocity=self.velocity(observed_latent,noisy,timesteps,context,context_lens=context_lens)
        mse=(velocity.float()-(noise-clean).float()).square().flatten(1).mean(1)
        image_mask=targets.image_valid.float()
        fm=(mse*self.scheduler.linear_timesteps_weights.to(clean.device)[idx]*image_mask).sum()/image_mask.sum().clamp_min(1)
        estimated_clean=noisy-s*velocity
        logits=self.point_head(estimated_clean)
        kp,valid_points=heatmap_loss(logits,targets,sigma)
        return dict(loss=fm+self.lambda_kp*kp,fm_loss=fm,kp_loss=kp,valid_points=valid_points,
                    clean_latent=estimated_clean,point_logits=logits,observed_latent=observed_latent,
                    velocity=velocity,noisy_latent=noisy,sigma=sigma)

    @torch.no_grad()
    def generate(self, observed, context, *, num_steps=20, seed=0,context_lens=None):
        if self.training:raise RuntimeError('goal generation requires eval()')
        if num_steps<1:raise ValueError('goal num_steps must be positive')
        observed_latent=self.codec.encode(observed)
        seeds = [int(seed)] * observed.shape[0] if isinstance(seed, int) else list(seed)
        if len(seeds) != observed.shape[0]:raise ValueError("one goal seed per sample required")
        z=torch.cat([torch.randn((1,*observed_latent.shape[1:]),device=observed.device,
            dtype=observed_latent.dtype,generator=torch.Generator(device=observed.device).manual_seed(int(s)))
            for s in seeds],dim=0)
        scheduler=FlowMatchScheduler(template='Wan');scheduler.set_timesteps(num_steps,shift=self.shift)
        for timestep in scheduler.timesteps:
            times=timestep.to(device=z.device,dtype=z.dtype).expand(z.shape[0])
            v=self.velocity(observed_latent,z,times,context,context_lens=context_lens)
            z=scheduler.step(v,timestep,z)
        return GoalPrediction(z,self.point_head(z),torch.ones(z.shape[0],80,device=z.device,dtype=torch.bool))

    def train(self, mode=True):
        return super().train(False if getattr(self, "_frozen", False) else mode)

    def train_stage_a(self):
        """Explicitly thaw owned weights for a new Stage A optimization run.

        Parent train() cannot thaw a deployed Stage B teacher. The borrowed
        VAE remains frozen and outside this module's parameter tree.
        """
        self._frozen = False
        self.requires_grad_(True)
        if isinstance(self.codec.vae, nn.Module):
            self.codec.vae.eval().requires_grad_(False)
        return self.train()

    def freeze(self):
        self._frozen = True
        self.eval().requires_grad_(False)
        return self
