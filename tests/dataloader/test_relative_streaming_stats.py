import numpy as np
from openwam.dataloader.relative_streaming_stats import StreamingStats


def test_exact_moments_and_masked_values():
    a=StreamingStats(2)
    x=np.array([[1.,3.],[2.,np.nan],[4.,9.]])
    a.update(x,np.array([[1,1],[1,0],[1,1]],bool))
    s=a.summary()
    np.testing.assert_array_equal(s['count'],[3,2])
    np.testing.assert_allclose(s['mean'],[7/3,6])
    np.testing.assert_allclose(s['std'],[np.std([1,2,4]),3])
    np.testing.assert_array_equal(s['min'],[1,3])
    np.testing.assert_array_equal(s['max'],[4,9])


def test_merge_and_persistence(tmp_path):
    rng=np.random.default_rng(42);x=rng.normal(size=(5000,3));m=rng.random(x.shape)>.2
    whole=StreamingStats(3);whole.update(x,m)
    left=StreamingStats(3);left.update(x[:2000],m[:2000])
    right=StreamingStats(3);right.update(x[2000:],m[2000:])
    right.save(tmp_path/'part.npz');left.merge(StreamingStats.load(tmp_path/'part.npz'))
    np.testing.assert_array_equal(left.hist,whole.hist)
    for k in whole.summary():np.testing.assert_allclose(left.summary()[k],whole.summary()[k],rtol=1e-12,atol=1e-12)


def test_quantile_bound_zero_negative_and_extremes():
    x=np.concatenate([-np.geomspace(1e-8,1e5,10000),np.zeros(1000),np.geomspace(1e-8,1e5,10000)])[:,None]
    a=StreamingStats(1);a.update(x,np.ones_like(x,bool))
    for q in [.01,.5,.99]:
        value,low,high=a.quantile(q);exact=np.quantile(x,q,axis=0)
        assert np.all(low<=exact) and np.all(exact<=high)
        assert np.all(np.abs(value-exact)<=(np.abs(exact)+a.scale)*a.relative_error)


def test_unavailable_dimension_and_invalid_input():
    import pytest
    a=StreamingStats(2);a.update(np.array([[1.,np.nan]]),np.array([[1,0]],bool))
    assert a.summary()['count'][1]==0
    with pytest.raises(ValueError):a.update(np.array([[1.,np.nan]]),np.ones((1,2),bool))
    with pytest.raises(ValueError):a.update(np.ones((1,2))*1e100,np.ones((1,2),bool))


def test_interpolated_quantile_crossing_zero_has_absolute_interval():
    x=np.array([-100.,-1.,.99902]+[100.]*148)[:,None]
    a=StreamingStats(1);a.update(x,np.ones_like(x,bool))
    value,low,high=a.quantile(.01);exact=np.quantile(x,.01,axis=0)
    assert np.all(low<=exact) and np.all(exact<=high)
    assert np.all(np.abs(value-exact)<=(high-low)/2+1e-15)
    # A relative-to-final-quantile bound would be incorrect near cancellation.
    assert abs(value[0]-exact[0])>a.relative_error*(abs(exact[0])+a.scale)
