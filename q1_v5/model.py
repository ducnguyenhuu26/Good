"""Grey-box semiparametric system identification and operator/program inference.

No simulator imports. Theta=[B(4), log d, log b, log kappa]. The shared learned
odd action response is a positive mixture of saturating tanh units. Stage
operators are learned from pairs with opaque program labels.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular
DT=.05
SIGMA=.0005

def to_xy(states, actions, delta):
    return np.c_[states,actions],delta

def psi(z,c):
    if isinstance(c,dict):
        slopes=np.asarray(c['slopes']);weights=np.asarray(c['weights']);
        return np.sum(weights*np.tanh(z[...,None]*slopes),axis=-1)/np.dot(weights,slopes)
    return z+c[0]*z**3+c[1]*z**5+c[2]*z**7

def dpsi(z,c):
    if isinstance(c,dict):
        slopes=np.asarray(c['slopes']);weights=np.asarray(c['weights']);
        return np.sum((weights*slopes)*(1-np.tanh(z[...,None]*slopes)**2),axis=-1)/np.dot(weights,slopes)
    return 1+3*c[0]*z*z+5*c[1]*z**4+7*c[2]*z**6

def response(x:np.ndarray,theta:np.ndarray,c:np.ndarray):
    B=theta[:4].reshape(2,2); rates=np.exp(theta[4:7]);v=x[:,2:4];q=x[:,:2]
    z=x[:,4:6]@B.T; p=psi(z,c)
    return DT*(p-rates[0]*v-rates[1]*np.linalg.norm(v,axis=1,keepdims=True)*v-rates[2]*q)

def jacobian(x,theta,c):
    B=theta[:4].reshape(2,2);rates=np.exp(theta[4:7]);v=x[:,2:4];q=x[:,:2];a=x[:,4:6]
    z=a@B.T;g=dpsi(z,c)
    n=len(x);J=np.zeros((n,2,7))
    for i in range(2):
        J[:,i,2*i:2*i+2]=DT*g[:,i,None]*a
        J[:,i,4]=-DT*rates[0]*v[:,i]
        J[:,i,5]=-DT*rates[1]*np.linalg.norm(v,axis=1)*v[:,i]
        J[:,i,6]=-DT*rates[2]*q[:,i]
    return J.reshape(2*n,7)

def fit_theta(x,y,c,guess=None,prior=None,max_nfev=80):
    if guess is None:guess=np.r_[np.eye(2).reshape(-1)*.9,np.log([.35,.12,.23])]
    if prior is None:pmean=np.zeros(7);psd=np.ones(7)*1e6
    else:pmean,psd=prior
    def fun(t):return np.r_[(response(x,t,c)-y).ravel()/SIGMA,(t-pmean)/psd]
    def jac(t):return np.vstack((jacobian(x,t,c)/SIGMA,np.diag(1/psd)))
    lower=np.r_[[-2.5]*4,[-6]*3];upper=np.r_[[2.5]*4,[1]*3]
    o=least_squares(fun,np.clip(guess,lower+1e-7,upper-1e-7),jac=jac,method='trf',bounds=(lower,upper),max_nfev=max_nfev,ftol=1e-8,gtol=1e-7)
    J=jacobian(x,o.x,c);prec=J.T@J/(SIGMA*SIGMA)+np.diag(1/psd**2)
    cov=np.linalg.inv(prec+np.eye(7)*1e-10)
    return o.x,cov,float(np.mean((response(x,o.x,c)-y)**2))

class ProgramModel:
    def __init__(self,c,opmap,prior_mean=None,prior_sd=None):
        self.c=c if isinstance(c,dict) else np.asarray(c,dtype=float);self.ops=opmap
        self.prior_mean=prior_mean;self.prior_sd=prior_sd
    def transform(self,t,program):
        new=np.array(t,copy=True)
        for opaque_code in program:
            op=self.ops[int(opaque_code)]
            if op['mechanism']=='action': new[:4]=(op['L']@new[:4].reshape(2,2)).ravel()
            else: new[4+op['index']]+=np.log(op['ratio'])
        return new
    def transform_jac(self,program):
        M=np.eye(7)
        for code in program:
            op=self.ops[int(code)]
            if op['mechanism']=='action':
                L=op['L'];step=np.eye(7);step[:4,:4]=np.kron(L,np.eye(2));M=step@M
        return M
    def old_posterior(self,x,y):
        prior=(self.prior_mean,self.prior_sd) if self.prior_mean is not None else None
        return fit_theta(x,y,self.c,prior=prior)
    def score_candidates(self,x,y,t_old,cov_old,programs,budgets):
        """Approximate marginal likelihood, Laplace linearization in old theta.

        Formula uses matrix determinant lemma, 7x7 Cholesky; all budgets share
        the same full uninterrupted adaptation trajectory prefixes.
        """
        ans={int(n):[] for n in budgets};sigma2=SIGMA*SIGMA
        for program in programs:
            t=self.transform(t_old,program);M=self.transform_jac(program)
            # numerical stability: don't invert the old covariance directly
            old=0.5*(cov_old+cov_old.T)+np.eye(7)*1e-12
            C=M@np.linalg.cholesky(old)
            pred=response(x,t,self.c).reshape(-1)
            jc=jacobian(x,t,self.c)@C
            for n in budgets:
                ny=2*int(n);r=y[:n].reshape(-1)-pred[:ny];U=jc[:ny]
                A=np.eye(7)+U.T@U/sigma2;chol=np.linalg.cholesky(A)
                z=U.T@r/sigma2
                v=solve_triangular(chol,z,lower=True)
                quad=max(0,float(r@r/sigma2-v@v))
                logdet=2*np.log(np.diag(chol)).sum()+ny*np.log(sigma2)
                score=.5*(quad+logdet+ny*np.log(2*np.pi))
                ans[int(n)].append((score,program,float(r@r/ny)))
        return {n:sorted(v,key=lambda z:z[0]) for n,v in ans.items()}

def joint_map(model,x0,y0,x1,y1,program,initial_theta=None,prior_mean=None,prior_sd=None):
    """Jointly fit pre-stage theta given both old and N adaptation samples.

    Program hypothesis is a candidate, never a known inference-time label.
    """
    if initial_theta is None:initial_theta=model.old_posterior(x0,y0)[0]
    if prior_mean is None:prior_mean=model.prior_mean
    if prior_sd is None:prior_sd=model.prior_sd
    if prior_mean is None:prior_mean=np.zeros(7);prior_sd=np.ones(7)*1e6
    M=model.transform_jac(program)
    def fun(t):
        after_t=model.transform(t,program)
        return np.r_[(response(x0,t,model.c)-y0).ravel()/SIGMA,
                     (response(x1,after_t,model.c)-y1).ravel()/SIGMA,
                     (t-prior_mean)/prior_sd]
    def jac(t):
        after_t=model.transform(t,program)
        return np.vstack((jacobian(x0,t,model.c)/SIGMA,
                          jacobian(x1,after_t,model.c)@M/SIGMA,
                          np.diag(1/prior_sd)))
    lower=np.r_[[-2.5]*4,[-6]*3];upper=np.r_[[2.5]*4,[1]*3]
    out=least_squares(fun,np.clip(initial_theta,lower+1e-7,upper-1e-7),jac=jac,bounds=(lower,upper),
                      method='trf',max_nfev=65,ftol=1e-8,gtol=1e-7)
    j=jac(out.x)
    cov=np.linalg.inv(j.T@j+np.eye(7)*1e-9)
    return out.x,cov,float(out.cost)

def normal_mixture_shrink(estimate:float, se:float, mean:float, sd:float,
                          *, broad_sd:float=.22, broad_weight:float=.12):
    import math
    denom=[];component=[]
    for variance,weight in ((sd**2,1-broad_weight),(broad_sd**2,broad_weight)):
        mvar=variance+se**2
        logp=math.log(weight)-.5*(math.log(mvar)+(estimate-mean)**2/mvar)
        mu=(estimate*variance+mean*se**2)/mvar
        denom.append(logp);component.append(mu)
    vmax=max(denom);probs=np.exp(np.asarray(denom)-vmax);probs/=sum(probs)
    return float(np.dot(probs,component)),float(probs[-1])
