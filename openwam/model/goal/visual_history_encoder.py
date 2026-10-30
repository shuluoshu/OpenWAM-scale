"""Frozen visual-history conditioner with the Stage1 target decoder removed."""
from torch import nn

from .history import HistoryConfig
from .visual_history import VisualHistoryGoalExpert


class FrozenVisualHistoryEncoder(VisualHistoryGoalExpert):
    """Retain precisely the modules used by ``condition_tokens``.

    HistoryConfig describes the ten input frame slots only. Its historical
    tokens_per_frame/compressor_dim fields do not describe this expert's pool:
    ``history_options`` stores its single total history-token budget instead.
    The returned memory remains 120 current spatial tokens.
    """
    def __init__(self, expert):
        nn.Module.__init__(self)
        self.text_dim = expert.context_width
        self.dit = expert.dit
        del self.dit.blocks
        del self.dit.head
        del self.dit.text_embedding
        self.codec = expert.codec
        self.dit_config = expert.dit_config
        self.shift = expert.shift
        self.history_options = expert.history_options
        self.history_pool = expert.history_pool
        self.visual_adapter = expert.visual_adapter
        self.history_gate = expert.history_gate
        self._history_cache = expert._history_cache
        self.history_config = HistoryConfig(enabled=True)
        self.observation_mode = 'multiview'
        self.num_tokens = 120
        self.token_dim = self.dit.dim
        self.use_gradient_checkpointing = False
        self.freeze()

    @property
    def context_width(self):
        return self.text_dim

    def forward(self, observed, **kwargs):
        return self.condition_tokens(observed, **kwargs)

    def generate(self, *args, **kwargs):
        raise RuntimeError('visual history Stage2 encoder has no goal decoder')
