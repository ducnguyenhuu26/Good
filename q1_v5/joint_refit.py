"""End-to-end observed-data re-estimation of operator magnitudes with shared task parameters."""
import numpy as np,torch

def joint_refit(c, before, after, thetas, ops, physical_to_opaque, iterations=220):
    torch.set_num_threads(2)
    x0=torch.as_tensor(np.stack([t[0] for t in before]),dtype=torch.float64)
    y0=torch.as_tensor(np.stack([t[1] for t in before]),dtype=torch.float64)
    xa=torch.as_tensor(np.stack([[after[i][b][0] for b in range(len(before))] for i in range(4)]),dtype=torch.float64)
    ya=torch.as_tensor(np.stack([[after[i][b][1] for b in range(len(before))] for i in range(4)]),dtype=torch.float64)
    theta=torch.nn.Parameter(torch.tensor(thetas,dtype=torch.float64))
    # Only optimize learned group operators and background coefficients (no simulator theta)
    L_raw={i:torch.nn.Parameter(torch.tensor(ops[physical_to_opaque[i]]['L'],dtype=torch.float64)) for i in range(4) if ops[physical_to_opaque[i]]['mechanism']=='action'}
    r_raw={i:torch.nn.Parameter(torch.tensor(np.log(ops[physical_to_opaque[i]]['ratio']),dtype=torch.float64)) for i in range(4) if ops[physical_to_opaque[i]]['mechanism']!='action'}
    parameters=[theta]+list(L_raw.values())+list(r_raw.values());optimizer=torch.optim.LBFGS(parameters,lr=.65,max_iter=iterations,line_search_fn='strong_wolfe',history_size=30,tolerance_grad=1e-10,tolerance_change=1e-12)
    slopes=torch.tensor(c['slopes'],dtype=torch.float64);weights=torch.tensor(c['weights'],dtype=torch.float64)
    def predict(X,T):
        B=T[...,:4].reshape(*T.shape[:-1],2,2)
        z=torch.einsum('...ij,...nj->...ni',B,X[...,4:6])
        psi=(weights*torch.tanh(z[...,None]*slopes)).sum(-1)/(weights*slopes).sum()
        rate=T[...,4:7].exp();v=X[...,2:4];q=X[...,0:2]
        return .05*(psi-rate[...,0,None,None]*v-rate[...,1,None,None]*torch.linalg.vector_norm(v,dim=-1,keepdim=True)*v-rate[...,2,None,None]*q)
    I=torch.eye(2,dtype=torch.float64)
    init_L={i:p.detach().clone() for i,p in L_raw.items()}
    counter=[0]
    def closure():
        optimizer.zero_grad()
        orig=((predict(x0,theta)-y0)/.0005).square().mean()
        nl=[]
        for i in range(4):
            t=theta.clone()
            if i in L_raw:
                t=torch.cat((torch.einsum('ij,bjk->bik',L_raw[i],theta[:,:4].reshape(-1,2,2)).reshape(-1,4),theta[:,4:7]),dim=1)
            else:
                j=ops[physical_to_opaque[i]]['index'];t=t+torch.nn.functional.one_hot(torch.tensor(4+j),7).double()[None,:]*r_raw[i]
            nl.append(((predict(xa[i],t)-ya[i])/.0005).square().mean())
        op_penalty=sum((p-init_L[i]).square().mean()*.00001 for i,p in L_raw.items())
        loss=.5*(orig+sum(nl))/5+op_penalty
        loss.backward();counter[0]+=1
        return loss
    optimizer.step(closure)
    out={k:dict(v) for k,v in ops.items()}
    for i,p in L_raw.items():out[physical_to_opaque[i]]['L']=p.detach().numpy()
    for i,p in r_raw.items():out[physical_to_opaque[i]]['ratio']=float(p.exp().detach())
    return out,theta.detach().numpy(),counter[0]
