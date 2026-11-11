import torch


def test_empty_runtime_cache_is_populated_by_validation():
    from scripts.diagnostics.check_goal_memory_gpu import run_expert
    from test_goal_memory_core import fixture
    from openwam.model.goal.goal_memory_cache import GoalMemoryRuntimeCache
    model, inputs = fixture()
    cache = GoalMemoryRuntimeCache(model_identity='test')
    inputs['memory_frame_ids'] = torch.tensor([[0,1],[0,1]])
    with torch.no_grad():
        run_expert(model,inputs,cache)
        assert cache.misses > 0
        run_expert(model,inputs,cache)
        assert cache.hits > 0
