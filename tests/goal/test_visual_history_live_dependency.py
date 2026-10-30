"""Frozen visual-history bundles validate the actual live text dependencies."""
import pytest
import torch
from openwam.model.goal import runtime
from test_visual_history import make_model
from openwam.model.goal.dependencies import dependency_identity
from openwam.train.goal_visual_history import save_visual_history_bundle


class LiveText(torch.nn.Module):
    def __init__(self,dim=16):
        super().__init__();self.dim=dim;self.projection=torch.nn.Linear(dim,dim)


class LiveTokenizer:
    def __init__(self):self.vocab={'move':0,'stop':1}
    def get_vocab(self):return self.vocab


@pytest.fixture
def fixture(tmp_path):
    expert=make_model().freeze();text=LiveText();tokenizer=LiveTokenizer()
    deps=dependency_identity(expert.codec.vae,text_dim=16,text_encoder=text,tokenizer=tokenizer)
    bundle=tmp_path/'visual.safetensors';save_visual_history_bundle(expert,bundle,dependency_identities=deps)
    def load(**overrides):
        args=dict(vae=expert.codec.vae,text_dim=16,text_encoder=text,tokenizer=tokenizer,
            device='cpu',dtype=torch.float32)
        args.update(overrides)
        return runtime.load_stage2_goal(bundle,**args)
    return text,tokenizer,deps,load


def test_live_dependencies_derive_verified_fingerprint(fixture):
    text,tokenizer,deps,load=fixture
    encoder=load()
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    assert not hasattr(encoder.dit,'head') and not hasattr(encoder.dit,'text_embedding')


@pytest.mark.parametrize('dependency',['encoder','tokenizer'])
def test_changed_live_dependency_cannot_be_hidden_by_explicit_fingerprint(fixture,dependency):
    text,tokenizer,deps,load=fixture
    if dependency=='encoder':text.projection.weight.data.add_(1)
    else:tokenizer.vocab['new']=2
    with pytest.raises(ValueError,match='fingerprint|text dependency'):
        load(precomputed_text_fingerprint=deps['text']['fingerprint'])


def test_explicit_fingerprint_must_match_live_identity(fixture):
    _,_,_,load=fixture
    with pytest.raises(ValueError,match='fingerprint|text dependency'):
        load(precomputed_text_fingerprint='c'*64)


def test_live_encoder_dimension_must_match_declared_context_width(fixture):
    text,_,deps,load=fixture;text.dim=32
    with pytest.raises(ValueError,match='dimension'):
        load(precomputed_text_fingerprint=deps['text']['fingerprint'])


def test_precomputed_only_identity_remains_supported(fixture):
    _,_,deps,load=fixture
    encoder=load(text_encoder=None,tokenizer=None,precomputed_text_fingerprint=deps['text']['fingerprint'])
    assert not encoder.training and not hasattr(encoder.dit,'head')
