"""Native Stage2 action expert with a single shared Stage1 memory/video expert."""
from dataclasses import replace
import inspect
import logging
import torch
from torch import nn
from openwam.model.goal.goal_memory_masks import view_token_mask


class GoalMemoryPolicy(nn.Module):
    def __init__(self, video, action_backbone, proprio_encoder, *, attention_mask_mode='isolated',
                 future_video_mode='none', lambda_keyframe=.1, freeze_video=False,
                 allow_approximate_low_precision_cache=False, goal_dropout=0.):
        super().__init__()
        if attention_mask_mode not in ('isolated','action_sees_video') or future_video_mode not in ('none','joint'):
            raise ValueError('invalid Stage2 routing mode')
        if attention_mask_mode=='action_sees_video' and future_video_mode!='joint':
            raise ValueError('action_sees_video requires joint future generation')
        if action_backbone.num_layers!=len(video.dit.blocks) or action_backbone.num_heads!=video.dit.blocks[0].num_heads or action_backbone.head_dim!=video.dit.dim//video.dit.blocks[0].num_heads:
            raise ValueError('native experts must share attention dimensions and depth')
        self.video,self.action_backbone,self.proprio_encoder=video,action_backbone,proprio_encoder
        self.attention_mask_mode,self.future_video_mode=attention_mask_mode,future_video_mode
        self.lambda_keyframe=float(lambda_keyframe);self.freeze_video=freeze_video
        self.allow_approximate_low_precision_cache=allow_approximate_low_precision_cache
        if not 0<=goal_dropout<=1:raise ValueError("goal_dropout must lie in [0,1]")
        self.goal_dropout=goal_dropout
        if freeze_video: self.video.requires_grad_(False)

    def _core(self, current_latents, memory_latents, text, noisy_target_latents=None, target_timestep=None, **kwargs):
        b=current_latents.shape[0]
        allowed=inspect.signature(self.video.prepare_state).parameters
        kw={k:v for k,v in kwargs.items() if k in allowed}
        kw['warmup_y_text']=False
        device_type = current_latents.device.type
        execution_dtype = (torch.get_autocast_dtype(device_type)
                           if torch.is_autocast_enabled(device_type) else current_latents.dtype)
        if execution_dtype!=torch.float32 and kw.get('runtime_cache') is not None:
            if not getattr(self,'_cache_mode_logged',False):
                logging.getLogger(__name__).warning('Low precision memory KV cache: %s',
                    'approximate opt-in enabled' if self.allow_approximate_low_precision_cache else 'disabled; FP32 exactness only')
                self._cache_mode_logged=True
            if not self.allow_approximate_low_precision_cache:kw.pop('runtime_cache',None)
        s=self.video.prepare_state(current_latents,memory_latents,
            torch.zeros_like(current_latents) if noisy_target_latents is None else noisy_target_latents,
            torch.zeros(b,device=current_latents.device) if target_timestep is None else target_timestep,text,**kw)
        if noisy_target_latents is None:
            end=s.target_slice.start
            s=replace(s,tokens=s.tokens[:,:end],time=s.time[:,:end],modulation=s.modulation[:,:end],
                frequencies=s.frequencies[:end],text_query_gate=s.text_query_gate[:,:end],
                key_valid=s.key_valid[:,:end],attention_mask=s.attention_mask[:,:,:end,:end],target_slice=slice(end,end))
        return s

    def _future(self, clean, latent, timestep, valid=None):
        d=self.video.dit;b,c,t,h,w=latent.shape
        if t!=2 or c!=48 or (h,w)!=self.video.config.latent_shape[-2:]: raise ValueError('future V shape must match native latent canvas')
        valid=torch.ones(b,t,3,dtype=torch.bool,device=latent.device) if valid is None else valid
        if valid.shape!=(b,t,3) or valid.dtype!=torch.bool: raise ValueError('future_view_valid must be boolean [B,T,3]')
        keys=view_token_mask(valid,h//2,w//2).flatten(1)
        pixels=keys.reshape(b,t,h//2,w//2).repeat_interleave(2,-2).repeat_interleave(2,-1)[:,None]
        latent=torch.where(pixels,latent,0.)
        if not torch.isfinite(latent).all(): raise ValueError('valid future pixels must be finite')
        tokens=d.patchify(latent).flatten(2).transpose(1,2);n=tokens.shape[1]
        time=self.video._time(timestep[:,None].expand(b,n),latent.dtype)
        freq=torch.cat([self.video._frequencies(len(clean.memory_slices)+2+i,h//2,w//2,latent.device) for i in range(t)])
        return replace(clean,tokens=tokens,time=time,modulation=d.time_projection(time).unflatten(-1,(6,d.dim)),
            frequencies=freq,key_valid=keys,text_query_gate=clean.context_mask.any(-1).reshape(b,1,1).expand(b,n,1),
            cached_frames=None,goal_kv_override=None,current_slice=slice(0,0),target_slice=slice(0,n),grid=(t,h//2,w//2)),pixels

    def _action(self, actions, timestep, text, proprio, text_mask=None):
        if proprio.shape!=(actions.shape[0],112) or actions.shape[1:]!=(32,112): raise ValueError('Stage2 requires proprio112 and action32x112')
        mask=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool) if text_mask is None else text_mask
        context=torch.cat((torch.where(mask[...,None],text,0.),self.proprio_encoder(proprio)[:,None]),1)
        mask=torch.cat((mask,torch.ones(mask.shape[0],1,device=mask.device,dtype=torch.bool)),1)
        return self.action_backbone.prepare_state(actions,timestep.to(actions),context=context,context_mask=mask)

    def _reader_valid(self,state,drop):
        valid=state.key_valid[:,:state.goal_slice.stop].clone()
        valid[:,state.goal_slice] &= ~drop[:,None]
        return valid

    def _attend(self,i,q,k,v,valid):
        return self.video.dit.blocks[i].self_attn.attn(q,k,v,attn_mask=valid)

    def forward(self,noisy_actions,action_timestep,*,current_latents,memory_latents,text,proprio,
                noisy_target_latents=None,target_timestep=None,noisy_future_latents=None,future_timestep=None,
                future_view_valid=None,**kwargs):
        s=self._core(current_latents,memory_latents,text,noisy_target_latents,target_timestep,**kwargs)
        f=None
        if noisy_future_latents is not None:
            if self.future_video_mode!='joint': raise ValueError('V supplied with generation disabled')
            f,_=self._future(s,noisy_future_latents,action_timestep if future_timestep is None else future_timestep,future_view_valid)
        if self.attention_mask_mode=='action_sees_video' and f is None: raise ValueError('joint inference requires noisy future V')
        a=self._action(noisy_actions,action_timestep,text,proprio,kwargs.get('text_mask'))
        drop=torch.rand(noisy_actions.shape[0],device=noisy_actions.device)<(self.goal_dropout if self.training or self.goal_dropout==1 else 0)
        for i in range(len(self.video.dit.blocks)):
            if self.training and self.video.config.use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint
                # Capture immutable per-layer containers; native post steps mutate.
                ss=replace(s);ff=None if f is None else replace(f)
                aa=replace(a,payload=replace(a.payload))
                def step(st,ft,at,layer=i,ss=ss,ff=ff,aa=aa):
                    ls=replace(ss,tokens=st)
                    lf=None if ff is None else replace(ff,tokens=ft)
                    la=replace(aa,payload=replace(aa.payload,x_action=at))
                    ls,lf,la=self._step(layer,ls,lf,la,drop)
                    return ls.tokens,ft if lf is None else lf.tokens,la.payload.x_action
                st,ft,at=checkpoint(step,s.tokens,s.tokens[:,:0] if f is None else f.tokens,a.payload.x_action,use_reentrant=False)
                s.tokens=st
                if f is not None:f.tokens=ft
                a.payload.x_action=at
            else:self._step(i,s,f,a,drop)
        out=dict(action_velocity=self.action_backbone.extract_prediction(a),goal_tokens=s.tokens[:,s.goal_slice])
        if noisy_target_latents is not None:out['keyframe_velocity']=self.video.outputs(s)['velocity']
        if f is not None:out['future_velocity']=self.video.dit.unpatchify(self.video.dit.head(f.tokens,f.time),f.grid)
        return out

    def _step(self,i,s,f,a,drop):
        q,k,v,p=self.video.pre_attn_at_layer(i,s);end=s.goal_slice.stop
        ck,cv,valid=k[:,:end],v[:,:end],s.key_valid[:,:end]
        if f is not None:
            fq,fk,fv,fp=self.video.pre_attn_at_layer(i,f)
            fo=self._attend(i,fq,torch.cat((ck,fk),1),torch.cat((cv,fv),1),torch.cat((valid,f.key_valid),1)[:,None,None])
        valid=self._reader_valid(s,drop)
        aq,ak,av,ap=self.action_backbone.pre_attn_at_layer(i,a)
        if self.attention_mask_mode=='action_sees_video':
            ck,cv,valid=torch.cat((ck,fk),1),torch.cat((cv,fv),1),torch.cat((valid,f.key_valid),1)
        ao=self._attend(i,aq,torch.cat((ck,ak),1),torch.cat((cv,av),1),torch.cat((valid,torch.ones(ak.shape[:2],device=ak.device,dtype=torch.bool)),1)[:,None,None])
        self.video.post_attn_at_layer(i,s,self._attend(i,q,k,v,s.attention_mask),p)
        if f is not None:self.video.post_attn_at_layer(i,f,fo,fp)
        self.action_backbone.post_attn_at_layer(i,a,ao,ap)
        return s,f,a

    def prefill(self,*,current_latents,memory_latents,text,**kwargs):
        if self.training or torch.is_grad_enabled():raise ValueError('prefill requires eval/no_grad')
        s=self._core(current_latents,memory_latents,text,**{k:v for k,v in kwargs.items() if k not in ('noisy_target_latents','target_timestep')})
        cache=[]
        drop=torch.full((current_latents.shape[0],),self.goal_dropout==1,device=current_latents.device,dtype=torch.bool)
        for i in range(len(self.video.dit.blocks)):
            q,k,v,p=self.video.pre_attn_at_layer(i,s)
            cache.append((k,v,self._reader_valid(s,drop)))
            self.video.post_attn_at_layer(i,s,self._attend(i,q,k,v,s.attention_mask),p)
        return cache

    def action_from_prefill(self,cache,noisy_actions,action_timestep,text,proprio,text_mask=None):
        if self.attention_mask_mode!='isolated':raise ValueError('cached action-only path requires isolated mode')
        a=self._action(noisy_actions,action_timestep,text,proprio,text_mask)
        for i,(k,v,valid) in enumerate(cache):
            q,ak,av,p=self.action_backbone.pre_attn_at_layer(i,a)
            mask=torch.cat((valid,torch.ones(ak.shape[:2],device=ak.device,dtype=torch.bool)),1)[:,None,None]
            self.action_backbone.post_attn_at_layer(i,a,self._attend(i,q,torch.cat((k,ak),1),torch.cat((v,av),1),mask),p)
        return self.action_backbone.extract_prediction(a)

    @torch.no_grad()
    def infer_actions(self,noise,*,steps=10,future_noise=None,**batch):
        if self.training or steps<1:raise ValueError('inference requires eval and positive steps')
        x=noise.clone();f=future_noise
        if (f is not None and batch.get('future_view_valid') is None
                and batch.get('observed_view_valid') is not None):
            batch['future_view_valid'] = batch['observed_view_valid'][:,None].expand(-1,f.shape[2],-1)
        cache=self.prefill(**batch) if self.attention_mask_mode=='isolated' else None
        for j in range(steps):
            t=torch.full((x.shape[0],),1000*(1-j/steps),device=x.device,dtype=x.dtype)
            if cache is not None:v=self.action_from_prefill(cache,x,t,batch['text'],batch['proprio'],batch.get('text_mask'))
            else:
                o=self(x,t,noisy_future_latents=f,future_timestep=t,**batch);v=o['action_velocity'];f=f-o['future_velocity']/steps
            x=x-v/steps
        return x

    def training_loss(self,*,actions,action_mask,proprio,current_latents,memory_latents,text,
                      target_latents=None,future_latents=None,future_view_valid=None,update=0,total_updates=1,
                      sigma=None,action_noise=None,**kwargs):
        b=actions.shape[0];device=actions.device
        sigma=torch.rand(b,device=device) if sigma is None else sigma
        if sigma.shape!=(b,) or not torch.isfinite(sigma).all() or ((sigma<0)|(sigma>1)).any():raise ValueError('invalid action sigma')
        if action_mask.dtype!=torch.bool or action_mask.shape!=actions.shape:raise ValueError('action_mask must match actions')
        clean=torch.where(action_mask,actions,0.)
        if not torch.isfinite(clean).all():raise ValueError('valid actions must be finite')
        noise=torch.randn_like(clean) if action_noise is None else action_noise
        noise=torch.where(action_mask,noise,0.)
        blend=sigma.to(clean)[:,None,None]
        noisy=(1-blend)*clean+blend*noise
        memory_valid=self.video._valid(kwargs.pop('memory_valid',None),memory_latents.shape[:2],device,'memory_valid')
        if self.training:
            memory_valid=memory_valid & (torch.rand(b,1,device=device)>=self.video.config.history_dropout)
            mix=self.video.config.history_noise_mix*torch.rand(*memory_latents.shape[:2],1,1,1,1,device=device,dtype=memory_latents.dtype)
            memory_latents=torch.sqrt(1-mix.square())*memory_latents+mix*torch.randn_like(memory_latents)
        extra={};targets={}
        for name,latent in [('keyframe',target_latents),('future',future_latents)]:
            if latent is None:continue
            if name=='future' and self.future_video_mode!='joint':raise ValueError('future targets forbidden in none mode')
            t=latent.shape[2];h,w=latent.shape[-2:]
            if name=='keyframe':
                views=self.video._valid(kwargs.get('target_view_valid'),(b,3),device,'target_view_valid')[:,None]
                views=views & self.video._valid(kwargs.get('target_image_valid'),(b,),device,'target_image_valid')[:,None,None]
            else:
                if t!=2:raise ValueError('future V must be two native temporal latents')
                views=self.video._valid(future_view_valid,(b,t,3),device,'future_view_valid')
            pixels=view_token_mask(views,h//2,w//2).reshape(b,t,h//2,w//2).repeat_interleave(2,-2).repeat_interleave(2,-1)[:,None]
            lc=torch.where(pixels,latent,0.)
            if not torch.isfinite(lc).all():raise ValueError('valid image targets must be finite')
            ln=torch.where(pixels,torch.randn_like(lc),0.);ls=torch.rand(b,device=device)
            target=ln-lc;blend=ls.to(lc)[:,None,None,None,None]
            nl=(1-blend)*lc+blend*ln
            key='target' if name=='keyframe' else 'future'
            extra['noisy_'+key+'_latents']=nl;extra[key+'_timestep']=ls*1000
            targets[name]=(target,pixels)
        if self.future_video_mode=='joint' and future_latents is None:raise ValueError('joint training requires future targets')
        out=self(noisy,sigma*1000,current_latents=current_latents,memory_latents=memory_latents,text=text,proprio=proprio,
                 memory_valid=memory_valid,future_view_valid=future_view_valid,**extra,**kwargs)
        error=torch.where(action_mask,out['action_velocity'].float()-(noise-clean).float(),0.)
        loss=error.square().sum()/action_mask.sum().clamp_min(1);out['action_loss']=loss
        zero=loss*0
        for name in ('keyframe','future'):
            fm=zero
            if name in targets:
                target,pixels=targets[name]
                error=torch.where(pixels,out[name+'_velocity'].float()-target.float(),0.)
                fm=error.square().sum()/(pixels.sum()*target.shape[1]).clamp_min(1)
            weight=self.lambda_keyframe if name=='keyframe' else 1.
            out[name+'_fm_loss']=fm.detach() if self.freeze_video or weight==0 else fm
            if not self.freeze_video:loss=loss+weight*fm
        out['loss']=loss
        return out
