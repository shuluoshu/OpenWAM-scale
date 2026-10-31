"""Mergeable full-population moments and bounded-error signed-log quantiles.

No sample reservoir and no clipping: every finite valid scalar increments its
histogram. Quantile intervals contain the empirical linearly interpolated
quantile. Missing dimensions stay missing until registry policy is applied.
``relative_error`` controls bin width, not relative error of an interpolated
quantile near zero: consumers must publish the returned absolute interval.
"""
from pathlib import Path
import os
import numpy as np


class StreamingStats:
    def __init__(self, width=52, *, bins=32768, relative_error=.001, scale=1e-6):
        if width<1 or bins<1 or not 0<relative_error<1 or scale<=0:
            raise ValueError('invalid histogram parameters')
        self.width=int(width);self.bins=int(bins);self.relative_error=float(relative_error);self.scale=float(scale)
        self.log_gamma=np.log1p(self.relative_error)
        self.hist=np.zeros((width,2*bins+1),np.uint64)
        self.count=np.zeros(width,np.uint64);self.mean=np.zeros(width);self.m2=np.zeros(width)
        self.minimum=np.full(width,np.inf);self.maximum=np.full(width,-np.inf)

    def update(self, values, valid):
        x=np.asarray(values);m=np.asarray(valid,bool)
        if x.shape!=m.shape or x.ndim<2 or x.shape[-1]!=self.width:raise ValueError('invalid statistic shape')
        x=x.reshape(-1,self.width);m=m.reshape(x.shape)
        for j in range(self.width):
            a=x[:,j][m[:,j]].astype(np.float64)
            if not len(a):continue
            if not np.isfinite(a).all():raise ValueError('nonfinite valid scalar')
            with np.errstate(over='raise',invalid='raise'):
                b=np.floor(np.log1p(np.abs(a)/self.scale)/self.log_gamma)
            if np.any(b>=self.bins):raise ValueError('histogram domain exceeded; no clipping permitted')
            indices=self.bins+np.sign(a).astype(np.int64)*(b.astype(np.int64)+1)
            self.hist[j]+=np.bincount(indices,minlength=2*self.bins+1).astype(np.uint64)
            n=len(a);mu=a.mean();delta=mu-self.mean[j];old=int(self.count[j]);total=old+n
            self.m2[j]+=np.square(a-mu).sum()+delta*delta*old*n/total
            self.mean[j]+=delta*n/total;self.count[j]=total
            self.minimum[j]=min(self.minimum[j],float(a.min()));self.maximum[j]=max(self.maximum[j],float(a.max()))

    def merge(self, other):
        if (self.width,self.bins,self.relative_error,self.scale)!=(other.width,other.bins,other.relative_error,other.scale):
            raise ValueError('incompatible histogram policies')
        a=self.count.astype(float);b=other.count.astype(float);n=a+b;den=np.maximum(n,1)
        delta=other.mean-self.mean
        self.m2+=other.m2+delta*delta*a*b/den;self.mean+=delta*b/den
        self.count+=other.count;self.hist+=other.hist
        self.minimum=np.minimum(self.minimum,other.minimum);self.maximum=np.maximum(self.maximum,other.maximum)

    def _bounds(self, index):
        if index==self.bins:return 0.,0.
        k=abs(index-self.bins)-1
        low=self.scale*np.expm1(k*self.log_gamma);high=self.scale*np.expm1((k+1)*self.log_gamma)
        return (low,high) if index>self.bins else (-high,-low)

    def quantile(self, q):
        if not 0<=q<=1:raise ValueError('quantile outside [0,1]')
        low=np.full(self.width,np.nan);high=low.copy()
        for j,n in enumerate(self.count):
            if not n:continue
            rank=q*(int(n)-1);r0=int(np.floor(rank));r1=int(np.ceil(rank));fraction=rank-r0
            cumulative=np.cumsum(self.hist[j]);i0=int(np.searchsorted(cumulative,r0+1));i1=int(np.searchsorted(cumulative,r1+1))
            l0,h0=self._bounds(i0);l1,h1=self._bounds(i1)
            # Observed extrema tighten edge bins and preserve exact constants.
            l0=max(l0,self.minimum[j]);l1=max(l1,self.minimum[j]);h0=min(h0,self.maximum[j]);h1=min(h1,self.maximum[j])
            low[j]=(1-fraction)*l0+fraction*l1;high[j]=(1-fraction)*h0+fraction*h1
        return (low+high)/2,low,high

    def summary(self):
        present=self.count>0
        return dict(count=self.count.copy(),min=np.where(present,self.minimum,np.nan),max=np.where(present,self.maximum,np.nan),
                    mean=np.where(present,self.mean,np.nan),std=np.where(present,np.sqrt(np.maximum(self.m2,0)/np.maximum(self.count,1)),np.nan),
                    q01=self.quantile(.01)[0],q99=self.quantile(.99)[0])

    def save(self, path):
        path=Path(path);temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f,width=self.width,bins=self.bins,relative_error=self.relative_error,scale=self.scale,
                hist=self.hist,count=self.count,mean=self.mean,m2=self.m2,minimum=self.minimum,maximum=self.maximum)
            f.flush();os.fsync(f.fileno())
        temp.replace(path)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            obj=cls(int(d['width']),bins=int(d['bins']),relative_error=float(d['relative_error']),scale=float(d['scale']))
            for name in ('hist','count','mean','m2','minimum','maximum'):setattr(obj,name,d[name].copy())
        if obj.hist.shape!=(obj.width,2*obj.bins+1) or not np.array_equal(obj.hist.sum(1),obj.count):
            raise ValueError('corrupt histogram counts')
        return obj
