import torch
from torch import nn

import torch.nn.functional as F

class sparseMoE(nn.Module):
    def __init__(self, expertModules, mode='ws'):
        super().__init__()
        self.experts    = nn.ModuleList(expertModules)
        #self.gate       = Gate(n_channel,len(self.experts), gumbel_tau=gumbel_tau)

        self.mode       = mode # [ 'single', 'moe',  'weight_sharing']
        self.register_load_state_dict_post_hook(lambda m,k : m._prepare_weight())
        self.is_single = len(self.experts)==1

    def _share_weight(self):
        # Copying Direction : 0 -> 1,2,3, ..
        if self.is_single: return

        ws_selector = self.experts[0].get_ws_selector()

        for i in range(1,len(self.experts)): 
            for w_key in ws_selector :
                module_path, _, param_name = w_key.rpartition(".")
                setattr(
                    self.experts[i].get_submodule(module_path), 
                    param_name, 
                    getattr(self.experts[0].get_submodule(module_path), param_name)
                    )


    def _prepare_weight(self):
        if self.mode == 'ws' :
            # Share Pointer
            self._share_weight()

        elif self.mode == 'moe':
            #Share hard copy of weight
            for i in range(1,len(self.experts)): 
                self.experts[i].load_state_dict(self.experts[0].state_dict())

    def forward(self, x,mask=None):
        if self.is_single or len(self.experts)==1:
            return self.experts[0](x)

        #if self.mode != 'single' and mask == None : 
            #mask = self.gate(x)

        #Random Mask
        #mask = F.one_hot(torch.randn(x.shape[0],len(self.experts)).argmax(dim=1),num_classes=len(self.experts)).float() # Noise-Free
        #mask = mask.to(x.device)


        mask,logit = mask
        self.mask = mask #For Bit loss Computing
        self.logit = logit

        outs = [ expert(x) for expert in self.experts ]
        out = torch.zeros_like(outs[0])
        for i in range(len(outs)):
            out += outs[i]*(mask[:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        return out

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_load_state_dict_post_hook(lambda m,k : m.load_parameter())

    def _copy_weight(self,a,b):
        # Load A from B
        a.load_state_dict(b.state_dict())

    def load_parameter(self):
        raise "NOT IMPLEMENTED"

    def get_ws_selector(self):
        # Fully-Qualifed String for torch.nn.Module
        # Eg ) conv1.0.weight
        raise "NOT IMPLEMENTED"

    def forward(self,x):
        raise "NOT IMPLEMENTED"
