"""Goal supervision stays outside deployment conditions.

Dataset targets are unbatched; collate_goal_targets adds the leading batch axis.
Visibility is -1 unknown, 0 occluded, 1 visible, independent of geometry.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Literal
import torch


@dataclass(frozen=True)
class GoalLabel:
    source_repo: str
    source_episode: int
    instruction_id: str
    anchor_source_frame: int
    goal_source_frame: int
    instruction_start: int
    instruction_end_exclusive: int
    success_verified: bool
    image_path: str
    calibration_id: str
    tcp_definition_id: str
    split: Literal['train', 'val', 'test']
    source_episode_length: int
    label_kind: Literal['reviewed_success', 'episode_end'] = 'reviewed_success'
    instruction: str = ''
    success_evidence: dict = field(default_factory=dict)
    baseline_seen_episode: bool = False
    review_status: str = 'pending_review'
    metadata: dict = field(default_factory=dict)

    @property
    def instruction_text(self):
        return self.instruction

    def to_dict(self):
        return asdict(self)


@dataclass
class GoalTargets:
    image: torch.Tensor
    image_valid: torch.Tensor
    uv: torch.Tensor  # Ordered left/right normalized head UV: pixels / [319, 255].
    point_valid: torch.Tensor
    in_frame: torch.Tensor
    visibility: torch.Tensor

    def to(self, device=None, dtype=None):
        return type(self)(**{f.name: getattr(self, f.name).to(device=device,
                            dtype=dtype if getattr(self, f.name).is_floating_point() else None)
                            for f in fields(self)})


def collate_goal_targets(targets):
    return GoalTargets(**{f.name: torch.stack([getattr(t, f.name) for t in targets])
                          for f in fields(GoalTargets)})


@dataclass
class GoalPrediction:
    latent: torch.Tensor
    point_logits: torch.Tensor
    token_valid: torch.Tensor


@dataclass
class GoalCondition:
    """Detached goal memory. Representation is bound by the loaded expert/adapter."""
    latent: torch.Tensor  # patch: [B,48,1,16,20]; query: [B,N1+N2,D]
    image_token_valid: torch.Tensor  # legacy field name; query includes both groups
    uv: torch.Tensor  # FP32 [B,2,2]; query runtime has no decoder, so NaN/unavailable
