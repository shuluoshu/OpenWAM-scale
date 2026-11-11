"""One shared native Wan VideoDiT for independent memory, current, G and Y.

The public pre/post split is the same native block computation used by Stage2.
Only G (and explicitly sampled warmup Y) receives text cross-attention. Memory
frames have independent self-attention and clean t=0, allowing exact per-frame
pre-RoPE KV reuse. Real ages are metadata; rotary time is window-relative.
"""
from dataclasses import dataclass, replace
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply, sinusoidal_embedding_1d
from .goal_memory_masks import group_attention_mask, view_token_mask
from .goal_memory_cache import tensor_fingerprint


@dataclass
class GoalMemoryConfig:
    num_goal_tokens: int = 32
    latent_shape: tuple = (48, 1, 24, 20)
    recent_frames: int = 2
    history_dropout: float = .15
    history_noise_mix: float = .1
    goal_dropout: float = 0.
    lambda_image: float = 1.
    lambda_uv: float = 1.
    warmup_fraction: float = .2
    warmup_y_text_probability: float = .5
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        self.latent_shape = tuple(self.latent_shape)
        if self.num_goal_tokens < 1 or not 0 <= self.recent_frames <= 4:
            raise ValueError('positive G count and zero to four recent memory frames required')
        if len(self.latent_shape) != 4 or self.latent_shape[:2] != (48, 1) or self.latent_shape[2] % 6 or self.latent_shape[3] % 4:
            raise ValueError('latent shape must be single-frame 48-channel three-view canvas')
        for name in ('history_dropout', 'history_noise_mix', 'goal_dropout', 'warmup_fraction', 'warmup_y_text_probability'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        if any(not math.isfinite(x) or x < 0 for x in (self.lambda_image, self.lambda_uv)):
            raise ValueError('loss weights must be finite nonnegative')


@dataclass
class GoalMemoryState:
    tokens: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    text_query_gate: torch.Tensor
    time: torch.Tensor
    modulation: torch.Tensor
    frequencies: torch.Tensor
    attention_mask: torch.Tensor
    key_valid: torch.Tensor
    memory_slices: list
    current_slice: slice
    goal_slice: slice
    target_slice: slice
    grid: tuple
    memory_ages: torch.Tensor
    memory_frame_ids: torch.Tensor
    cached_frames: object = None
    goal_kv_override: object = None


class GoalMemoryExpert(nn.Module):
    def __init__(self, dit, config=None, *, dit_config=None):
        super().__init__()
        self.config = config or GoalMemoryConfig()
        self.dit = dit
        self.dit_config = dict(dit_config or {})
        if dit.in_dim != 48 or tuple(dit.patch_size) != (1,2,2) or dit.has_image_input or dit.head.head.out_features != 192:
            raise ValueError('goal_memory requires native 48-channel Wan with (1,2,2) patches and no CLIP input')
        self.goal_queries = nn.Parameter(torch.randn(1, self.config.num_goal_tokens, dit.dim) / math.sqrt(dit.dim))
        self.horizon_embedding = nn.Sequential(nn.Linear(dit.freq_dim, dit.dim), nn.SiLU(), nn.Linear(dit.dim, dit.dim))
        self.uv_head = nn.Sequential(nn.LayerNorm(dit.dim), nn.Linear(dit.dim, dit.dim), nn.SiLU(), nn.Linear(dit.dim, 4))

    @property
    def context_width(self):
        return self.dit.text_embedding[0].in_features

    def _frequencies(self, frame, rows, columns, device):
        d = self.dit
        return torch.cat((d.freqs[0][frame, None, None].expand(rows, columns, -1),
                          d.freqs[1][:rows, None].expand(rows, columns, -1),
                          d.freqs[2][None, :columns].expand(rows, columns, -1)), -1).reshape(rows*columns, 1, -1).to(device)

    def _time(self, values, dtype):
        return self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, values.flatten()).to(dtype)).reshape(*values.shape, self.dit.dim)

    def _valid(self, value, shape, device, name):
        if value is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        value = torch.as_tensor(value, device=device)
        if value.shape != shape or value.dtype != torch.bool:
            raise ValueError(f'{name} must be boolean {shape}')
        return value

    def _sanitize(self, latent, token_valid, rows, columns):
        mask = token_valid.reshape(*token_valid.shape[:-1], rows, columns).repeat_interleave(2,-2).repeat_interleave(2,-1)
        safe = torch.where(mask[..., None, None, :, :], latent, 0.)
        if not torch.isfinite(safe).all():
            raise ValueError('valid latent regions must be finite')
        return safe

    def prepare_state(self, current_latents, memory_latents, noisy_target_latents, timestep, text, *,
                      text_mask=None, memory_valid=None, observed_view_valid=None, memory_view_valid=None,
                      memory_ages=None, memory_frame_ids=None, goal_horizon_s=None, target_image_valid=None, target_view_valid=None,
                      warmup_y_text=False, goal_tokens_override=None, goal_kv_override=None, runtime_cache=None, episode_id=None):
        cfg,d = self.config,self.dit
        b = current_latents.shape[0]
        shape = (b,*cfg.latent_shape)
        if tuple(current_latents.shape) != shape or tuple(noisy_target_latents.shape) != shape:
            raise ValueError(f'current and target latents must be {shape}')
        if memory_latents.ndim != 6 or memory_latents.shape[0] != b or tuple(memory_latents.shape[2:]) != cfg.latent_shape:
            raise ValueError('memory must be [B,M,48,1,H,W] independent frames')
        m = memory_latents.shape[1]
        if m > cfg.recent_frames+1:
            raise ValueError('memory exceeds configured sink plus recent window')
        device,dtype = current_latents.device,current_latents.dtype
        rows,columns = cfg.latent_shape[-2]//2,cfg.latent_shape[-1]//2
        n = rows*columns
        cv = view_token_mask(self._valid(observed_view_valid,(b,3),device,'observed_view_valid'),rows,columns)
        mv = view_token_mask(self._valid(memory_view_valid,(b,m,3),device,'memory_view_valid'),rows,columns)
        mv = mv & self._valid(memory_valid,(b,m),device,'memory_valid')[...,None]
        yv = view_token_mask(self._valid(target_view_valid,(b,3),device,'target_view_valid'),rows,columns)
        yv = yv & self._valid(target_image_valid,(b,),device,'target_image_valid')[:,None]
        current_latents = self._sanitize(current_latents,cv,rows,columns)
        memory_latents = self._sanitize(memory_latents,mv,rows,columns)
        noisy_target_latents = self._sanitize(noisy_target_latents,yv,rows,columns)
        patch = lambda x: d.patchify(x).flatten(2).transpose(1,2)
        c,y = patch(current_latents),patch(noisy_target_latents)
        memory = patch(memory_latents.reshape(b*m,*cfg.latent_shape)).reshape(b,m*n,d.dim) if m else c[:,:0]
        g = self.goal_queries.expand(b,-1,-1).to(dtype)
        if goal_tokens_override is not None:
            if goal_tokens_override.shape != g.shape or not torch.isfinite(goal_tokens_override).all():
                raise ValueError('goal override must be finite [B,G,D]')
            g = goal_tokens_override.to(g)
        tokens = torch.cat((memory,c,g,y),1)
        memory_slices = [slice(i*n,(i+1)*n) for i in range(m)]
        cs,gs,ys = slice(m*n,(m+1)*n),slice((m+1)*n,(m+1)*n+g.shape[1]),slice((m+1)*n+g.shape[1],tokens.shape[1])
        gv = torch.ones(b,g.shape[1],dtype=torch.bool,device=device)
        if self.training and cfg.goal_dropout:
            gv = gv & (torch.rand(b,1,device=device) >= cfg.goal_dropout)
        key_valid = torch.cat((mv.flatten(1),cv,gv,yv),1)
        attention_mask = group_attention_mask(memory_slices,cs,gs,ys,key_valid)
        timestep = torch.as_tensor(timestep,device=device)
        if timestep.shape != (b,) or not torch.isfinite(timestep).all():
            raise ValueError('diffusion timestep must be finite [B]')
        t = torch.zeros(b,tokens.shape[1],device=device,dtype=dtype)
        t[:,ys] = timestep[:,None]
        time = self._time(t,dtype)
        horizon = torch.zeros(b,device=device,dtype=dtype) if goal_horizon_s is None else torch.as_tensor(goal_horizon_s,device=device,dtype=dtype)
        if horizon.shape != (b,) or not torch.isfinite(horizon).all() or (horizon<0).any():
            raise ValueError('goal horizon must be finite nonnegative [B] seconds')
        time = torch.cat((time[:,:ys.start], time[:,ys] + self.horizon_embedding(sinusoidal_embedding_1d(d.freq_dim,horizon).to(dtype))[:,None]),1)
        modulation = d.time_projection(time).unflatten(-1,(6,d.dim))
        frequencies = [self._frequencies(i,rows,columns,device) for i in range(m+1)]
        gf = torch.cat((d.freqs[0][m+1],d.freqs[1][0],d.freqs[2][0]),-1)[None,None].expand(g.shape[1],1,-1).to(device)
        frequencies = torch.cat((*frequencies,gf,self._frequencies(m+2,rows,columns,device)),0)
        if text.ndim != 3 or text.shape[0] != b or text.shape[-1] != self.context_width:
            raise ValueError('text must be [B,L,text_dim]')
        valid_text = self._valid(text_mask,(b,text.shape[1]),device,'text_mask')
        text = torch.where(valid_text[...,None],text,0.)
        if not torch.isfinite(text).all():
            raise ValueError('valid text must be finite')
        if text.shape[1] == 0:
            text = torch.zeros(b,1,self.context_width,device=device,dtype=dtype)
            valid_text = torch.zeros(b,1,device=device,dtype=torch.bool)
        context = d.text_embedding(text.to(dtype))
        gate = torch.zeros(b,tokens.shape[1],1,device=device,dtype=torch.bool)
        gate[:,gs] = True
        warmup = torch.as_tensor(warmup_y_text,device=device,dtype=torch.bool)
        if warmup.ndim == 0: warmup = warmup.expand(b)
        if warmup.shape != (b,): raise ValueError('warmup_y_text must be bool or [B]')
        gate[:,ys] = warmup[:,None,None]
        gate = gate & valid_text.any(-1)[:,None,None]
        ages = torch.zeros(b,m,device=device) if memory_ages is None else torch.as_tensor(memory_ages,device=device)
        ids = torch.arange(m,device=device).expand(b,-1) if memory_frame_ids is None else torch.as_tensor(memory_frame_ids,device=device)
        if ages.shape != (b,m) or ids.shape != (b,m):
            raise ValueError('memory ages and frame IDs must be [B,M]')
        ages=torch.where(mv.any(-1),ages,0.)
        ids=torch.where(mv.any(-1),ids,0)
        if not torch.isfinite(ages).all() or not torch.isfinite(ids).all():
            raise ValueError('valid memory ages and frame IDs must be finite')
        state = GoalMemoryState(tokens,context,valid_text[:,None,None],gate,time,modulation,frequencies,
                                attention_mask,key_valid,memory_slices,cs,gs,ys,(1,rows,columns),ages,ids)
        state.goal_kv_override = goal_kv_override
        if runtime_cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError('runtime memory caching requires eval() and no_grad()')
            if memory_frame_ids is None:
                raise ValueError('runtime cache requires explicit source frame IDs')
            runtime_cache.set_episode(episode_id)
            runtime_cache.bind_model(self)
            state.cached_frames = []
            for j in range(m):
                batch_entries=[]
                for bi in range(b):
                    key=(runtime_cache.model_identity,int(ids[bi,j]),tensor_fingerprint(memory_latents[bi,j]),tensor_fingerprint(mv[bi,j]))
                    entry=runtime_cache.get(key)
                    if entry is None:
                        entry=self._encode_memory_frame(memory[bi:bi+1,j*n:(j+1)*n],mv[bi:bi+1,j],rows,columns)
                        runtime_cache.put(key,entry)
                    batch_entries.append(entry)
                state.cached_frames.append(batch_entries)
        return state

    def _encode_memory_frame(self,tokens,valid,rows,columns):
        time=self._time(torch.zeros(tokens.shape[:2],device=tokens.device,dtype=tokens.dtype),tokens.dtype)
        modulation=self.dit.time_projection(time).unflatten(-1,(6,self.dit.dim))
        freq=self._frequencies(0,rows,columns,tokens.device)
        layers=[]
        for block in self.dit.blocks:
            shifts=(block.modulation.to(modulation)+modulation).unbind(2)
            sm,sc,gm,sf,scf,gf=shifts
            x=modulate(block.norm1(tokens),sm,sc)
            q,k,v=block.self_attn.norm_q(block.self_attn.q(x)),block.self_attn.norm_k(block.self_attn.k(x)),block.self_attn.v(x)
            attention=block.self_attn.attn(rope_apply(q,freq,block.num_heads),rope_apply(k,freq,block.num_heads),v,attn_mask=valid[:,None,None])
            tokens=block.gate(tokens,gm,block.self_attn.o(attention))
            tokens=block.gate(tokens,gf,block.ffn(modulate(block.norm2(tokens),sf,scf)))
            layers.append((k,v,tokens))
        return layers

    def pre_attn_at_layer(self,layer_id,state):
        block=self.dit.blocks[layer_id]
        start=state.current_slice.start if state.cached_frames is not None else 0
        sm,sc,gm,sf,scf,gf=(block.modulation.to(state.modulation)+state.modulation[:,start:]).unbind(2)
        x=modulate(block.norm1(state.tokens[:,start:]),sm,sc)
        q=block.self_attn.norm_q(block.self_attn.q(x))
        k=block.self_attn.norm_k(block.self_attn.k(x));v=block.self_attn.v(x)
        if start:
            raw_k=torch.cat([torch.cat([entry[layer_id][0] for entry in frame],0) for frame in state.cached_frames],1)
            raw_v=torch.cat([torch.cat([entry[layer_id][1] for entry in frame],0) for frame in state.cached_frames],1)
            k=torch.cat((raw_k,k),1);v=torch.cat((raw_v,v),1)
            q=torch.cat((torch.zeros_like(raw_k),q),1)
        post=dict(gate_msa=gm,shift_mlp=sf,scale_mlp=scf,gate_mlp=gf,start=start,raw_k=k)
        q=rope_apply(q,state.frequencies,block.num_heads)
        k=rope_apply(k,state.frequencies,block.num_heads)
        if state.goal_kv_override is not None and layer_id in state.goal_kv_override:
            fixed_k,fixed_v=state.goal_kv_override[layer_id]
            expected=k[:,state.goal_slice].shape
            if fixed_k.shape != expected or fixed_v.shape != expected or not torch.isfinite(fixed_k).all() or not torch.isfinite(fixed_v).all():
                raise ValueError('G KV overrides must be finite native rotated [B,G,D] pairs')
            k=torch.cat((k[:,:state.goal_slice.start],fixed_k.to(k),k[:,state.goal_slice.stop:]),1)
            v=torch.cat((v[:,:state.goal_slice.start],fixed_v.to(v),v[:,state.goal_slice.stop:]),1)
        return q,k,v,post

    def post_attn_at_layer(self,layer_id,state,attn_output,post):
        block=self.dit.blocks[layer_id];start=post['start']
        x=block.gate(state.tokens[:,start:],post['gate_msa'],block.self_attn.o(attn_output[:,start:]))
        ca=block.cross_attn(block.norm3(x),state.context,ctx_mask=state.context_mask)
        # Gating AFTER output projection is essential: the native o bias is nonzero.
        x=x+torch.where(state.text_query_gate[:,start:],ca,0.)
        x=block.gate(x,post['gate_mlp'],block.ffn(modulate(block.norm2(x),post['shift_mlp'],post['scale_mlp'])))
        if start:
            memory=torch.cat([torch.cat([entry[layer_id][2] for entry in frame],0) for frame in state.cached_frames],1)
            x=torch.cat((memory,x),1)
        state.tokens=x
        return state

    def run_state(self,state):
        for i,block in enumerate(self.dit.blocks):
            def forward(tokens,layer=i):
                local=replace(state,tokens=tokens)
                q,k,v,post=self.pre_attn_at_layer(layer,local)
                start=post['start']
                attention=self.dit.blocks[layer].self_attn.attn(q[:,start:],k,v,attn_mask=local.attention_mask[:,:,start:])
                if start:
                    attention=torch.cat((torch.zeros_like(q[:,:start]),attention),1)
                return self.post_attn_at_layer(layer,local,attention,post).tokens
            if self.training and self.config.use_gradient_checkpointing:
                state.tokens=checkpoint(forward,state.tokens,use_reentrant=False)
            else:
                state.tokens=forward(state.tokens)
        return state

    def outputs(self,state):
        g=state.tokens[:,state.goal_slice]
        patches=self.dit.head(state.tokens[:,state.target_slice],state.time[:,state.target_slice])
        return dict(goal_tokens=g,velocity=self.dit.unpatchify(patches,state.grid),uv=self.uv_head(g.mean(1)).reshape(-1,2,2).float())

    def warmup_probability(self,update,total_updates):
        if total_updates < 1 or update < 0:
            raise ValueError('updates must be nonnegative with positive total_updates')
        span=self.config.warmup_fraction*total_updates
        return self.config.warmup_y_text_probability*max(0.,1.-update/span) if span else 0.

    def training_loss(self,current_latents,memory_latents,target_latents,text,target_uv,*,
                      target_image_valid,target_point_valid,target_in_frame,target_view_valid=None,update=0,total_updates=1,
                      sigma=None,noise=None,**prepare_kwargs):
        b=current_latents.shape[0];device=current_latents.device;cfg=self.config
        image_valid=self._valid(target_image_valid,(b,),device,'target_image_valid')
        rows,columns=cfg.latent_shape[-2]//2,cfg.latent_shape[-1]//2
        target_views=self._valid(target_view_valid,(b,3),device,'target_view_valid')
        target_tokens=view_token_mask(target_views,rows,columns) & image_valid[:,None]
        clean=self._sanitize(target_latents,target_tokens,rows,columns)
        pixel_valid=target_tokens.reshape(b,rows,columns).repeat_interleave(2,-2).repeat_interleave(2,-1)[:,None,None]
        sigma=torch.rand(b,device=device) if sigma is None else torch.as_tensor(sigma,device=device)
        if sigma.shape != (b,) or not torch.isfinite(sigma).all() or ((sigma<0)|(sigma>1)).any():
            raise ValueError('sigma must be finite [B] in [0,1]')
        noise=torch.randn_like(clean) if noise is None else noise.to(clean)
        if noise.shape != clean.shape: raise ValueError('noise must match target')
        noise=self._sanitize(noise,target_tokens,rows,columns)
        s=sigma.to(clean)[:,None,None,None,None]
        noisy=(1-s)*clean+s*noise
        mv=self._valid(prepare_kwargs.pop('memory_valid',None),memory_latents.shape[:2],device,'memory_valid')
        if self.training:
            mv=mv & (torch.rand(b,1,device=device)>=cfg.history_dropout)
            if cfg.history_noise_mix:
                mix=cfg.history_noise_mix*torch.rand(*memory_latents.shape[:2],1,1,1,1,device=device,dtype=memory_latents.dtype)
                memory_latents=torch.sqrt(1-mix.square())*memory_latents+mix*torch.randn_like(memory_latents)
        prepare_kwargs.setdefault('warmup_y_text',torch.rand(b,device=device)<self.warmup_probability(update,total_updates) if self.training else False)
        state=self.prepare_state(current_latents,memory_latents,noisy,sigma*1000,text,
            memory_valid=mv,target_image_valid=image_valid,target_view_valid=target_views,**prepare_kwargs)
        out=self.outputs(self.run_state(state))
        image_error=torch.where(pixel_valid,out['velocity'].float()-(noise-clean).float(),0.)
        fm=image_error.square().sum()/(pixel_valid.sum()*clean.shape[1]).clamp_min(1)
        if target_uv.shape != (b,2,2): raise ValueError('target UV must be [B,2,2]')
        pv=self._valid(target_point_valid,(b,2),device,'target_point_valid') & self._valid(target_in_frame,(b,2),device,'target_in_frame')
        pv=pv & image_valid[:,None] & target_views[:,0,None] & torch.isfinite(target_uv).all(-1)
        safe_uv=torch.where(pv[...,None],target_uv,0.)
        uv_error=torch.where(pv[...,None],out['uv']-safe_uv,0.)
        uv=uv_error.square().sum()/(2*pv.sum().clamp_min(1))
        return out|dict(loss=cfg.lambda_image*fm+cfg.lambda_uv*uv,fm_loss=fm,uv_loss=uv,sigma=sigma,
                       target_latent=clean,noisy_latent=noisy,clean_latent=noisy-s*out['velocity'],valid_points=pv.sum())

    forward=training_loss
