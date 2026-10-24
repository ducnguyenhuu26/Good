"""Uniform full mechanism operators and exact affine program composition."""
from __future__ import annotations
import torch
from torch import nn

class MechanismOperatorBank(nn.Module):
    def __init__(self,codes:int=4,dtype=torch.float64,device=None,selective:bool=False):
        super().__init__();self.codes=codes
        self.selective=bool(selective)
        eye=torch.eye(2,dtype=dtype,device=device).expand(codes,2,2).clone()
        if self.selective:
            self.L_delta=nn.Parameter(1e-3*torch.randn_like(eye))
            self.alpha_delta=nn.Parameter(1e-3*torch.randn(codes,3,dtype=dtype,device=device))
            # Rows are opaque codes; columns are action, linear drag, speed drag,
            # and position feedback. The data, not code identity, learns this map.
            self.assignment_logits=nn.Parameter(torch.zeros(codes,4,dtype=dtype,device=device))
        else:
            self.L=nn.Parameter(eye+1e-3*torch.randn_like(eye))
            self.alpha=nn.Parameter(torch.zeros(codes,3,dtype=dtype,device=device))
    @property
    def assignment_probabilities(self):
        if not self.selective:
            reference=self._parameters['alpha']
            return torch.ones((self.codes,4),dtype=reference.dtype,device=reference.device)
        return torch.softmax(self.assignment_logits,dim=-1)
    def effective_parameters(self):
        if not self.selective:return self._parameters['L'],self._parameters['alpha']
        eye=torch.eye(2,dtype=self.L_delta.dtype,device=self.L_delta.device)
        return (eye+self.assignment_probabilities[:,0,None,None]*self.L_delta,
            self.assignment_probabilities[:,1:]*self.alpha_delta)
    def transform(self,theta,code:int):
        L,alpha=self.effective_parameters()
        B=theta[..., :4].reshape(*theta.shape[:-1],2,2)
        changed=L[code]@B
        return torch.cat((changed.reshape(*theta.shape[:-1],4),theta[...,4:7]+alpha[code]),dim=-1)
    def affine(self,program):
        """Return theta' = M theta+c for listed application order."""
        L_codes,alpha_codes=self.effective_parameters();dtype=L_codes.dtype;device=L_codes.device
        L=torch.eye(2,dtype=dtype,device=device);alpha=torch.zeros(3,dtype=dtype,device=device)
        for code in program:
            L=L_codes[int(code)]@L
            alpha=alpha+alpha_codes[int(code)]
        M=torch.zeros(7,7,dtype=dtype,device=device)
        # Row-major vec(L B) = kron(L, I) vec(B).
        M[:4,:4]=torch.kron(L,torch.eye(2,dtype=dtype,device=device));M[4:,4:]=torch.eye(3,dtype=dtype,device=device)
        c=torch.cat((torch.zeros(4,dtype=dtype,device=device),alpha))
        return M,c
    def apply_program(self,theta,program):
        M,c=self.affine(program);return torch.einsum('ij,...j->...i',M,theta)+c
    def sparsity_penalty(self):
        L,alpha=self.effective_parameters()
        groups=[L-torch.eye(2,dtype=L.dtype,device=L.device),alpha[:,0:1],alpha[:,1:2],alpha[:,2:3]]
        return sum(torch.sqrt(g.square().sum(dim=tuple(range(1,g.ndim)))+1e-12).sum() for g in groups)
    def selectivity_penalty(self):
        if not self.selective:return torch.zeros((),dtype=self._parameters['alpha'].dtype,device=self._parameters['alpha'].device)
        p=self.assignment_probabilities.clamp_min(1e-12)
        # Low-entropy assignments encourage each opaque code to focus on a
        # small mechanism subset while leaving the selected group data-driven.
        return -(p*p.log()).sum(-1).mean()
