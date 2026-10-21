import numpy as np
import torch

from rwt.clsm.dynamics import ExecutableLaw
from rwt.clsm.schemas import EnvSchema
from rwt.clsm.train import train_reference


def test_reference_train_and_normalizer_roundtrip():
    schema = EnvSchema("toy",3,2,(1,2),(0,1),np.eye(2),np.ones(2),-np.ones(2),np.ones(2),.01)
    torch.manual_seed(8)
    law = ExecutableLaw(schema,hidden_dim=8)
    obs = torch.randn((24,3),dtype=torch.float64)
    action = torch.randn((24,2),dtype=torch.float64)
    target = obs + .1
    curve = train_reference(law,obs,action,target,epochs=8,lr=.01,seed=4)
    assert curve[-1] < curve[0]
    restored = ExecutableLaw(schema,hidden_dim=8)
    restored.load_state_dict(law.state_dict())
    z = torch.zeros(24,dtype=torch.long); theta=torch.zeros((24,3),dtype=torch.float64)
    assert torch.equal(law(obs,action,z,theta)[0],restored(obs,action,z,theta)[0])
    assert torch.equal(law.delta_scale,restored.delta_scale)
