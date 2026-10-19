"""Explicit Stage2 optimizer ownership; frozen dependencies never enter groups."""
G1_LRS = {'video': 1e-05, 'action': 0.0001, 'projection': 0.001, 'goal_adapter': 0.0001}

def build_g1_parameter_groups(architecture):
    """Assign each trainable parameter exactly once; reject unknown ownership.

    Inspect aliases as well: sharing weights between baseline and adapter groups
    is a configuration error, not a reason to silently pick one learning rate.
    """
    buckets = {name: [] for name in G1_LRS}
    owners = {}
    for name, parameter in architecture.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        if name.startswith(('action_backbone.action_encoder.', 'action_backbone.action_decoder.')):
            group = 'projection'
        elif name.startswith('action_backbone.'):
            group = 'action'
        elif name.startswith(('video_backbone.', 'proprio_encoder.')):
            group = 'video'
        elif name.startswith('goal_adapter.'):
            group = 'goal_adapter'
        else:
            raise ValueError(f'G1 has unexpected trainable parameter {name}; goal expert/point head must be frozen')
        identity = id(parameter)
        if identity in owners:
            if owners[identity] != group:
                raise ValueError(f'Parameter {name} is shared across optimizer groups {owners[identity]} and {group}')
            continue
        owners[identity] = group
        buckets[group].append(parameter)
    expected = {id(p) for p in architecture.parameters() if p.requires_grad}
    if set(owners) != expected:
        raise ValueError('G1 optimizer does not cover every trainable parameter exactly once')
    return [{'name': name, 'params': params, 'lr': G1_LRS[name]}
            for name, params in buckets.items() if params]

