"""Uniform full mechanism operators and exact affine program composition."""
from __future__ import annotations
import torch
from torch import nn

class MechanismOperatorBank(nn.Module):
    def __init__(self,codes:int=4,dtype=torch.float64,device=None):
        super().__init__();self.codes=codes
        eye=torch.eye(2,dtype=dtype,device=device).expand(codes,2,2).clone()
        self.L=nn.Parameter(eye+1e-3*torch.randn_like(eye))
        self.alpha=nn.Parameter(torch.zeros(codes,3,dtype=dtype,device=device))
    def transform(self,theta,code:int):
        B=theta[..., :4].reshape(*theta.shape[:-1],2,2)
        changed=self.L[code]@B
        return torch.cat((changed.reshape(*theta.shape[:-1],4),theta[...,4:7]+self.alpha[code]),dim=-1)
    def affine(self,program):
        """Return theta' = M theta+c for listed application order."""
        dtype=self.L.dtype;device=self.L.device
        L=torch.eye(2,dtype=dtype,device=device);alpha=torch.zeros(3,dtype=dtype,device=device)
        for code in program:
            L=self.L[int(code)]@L
            alpha=alpha+self.alpha[int(code)]
        M=torch.zeros(7,7,dtype=dtype,device=device)
        # Row-major vec(L B) = kron(L, I) vec(B).
        M[:4,:4]=torch.kron(L,torch.eye(2,dtype=dtype,device=device));M[4:,4:]=torch.eye(3,dtype=dtype,device=device)
        c=torch.cat((torch.zeros(4,dtype=dtype,device=device),alpha))
        return M,c
    def apply_program(self,theta,program):
        M,c=self.affine(program);return torch.einsum('ij,...j->...i',M,theta)+c
    def sparsity_penalty(self):
        groups=[self.L-torch.eye(2,dtype=self.L.dtype,device=self.L.device),self.alpha[:,0:1],self.alpha[:,1:2],self.alpha[:,2:3]]
        return sum(torch.sqrt(g.square().sum(dim=tuple(range(1,g.ndim)))+1e-12).sum() for g in groups)
