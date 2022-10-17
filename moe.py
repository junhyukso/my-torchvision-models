import torch
from torch import nn

class sparseMoE(nn.Module):
    def __init__(self, expertModules, mode='ws'):
        super().__init__()
        self.experts    = nn.ModuleList(expertModules)
        #self.gate       = Gate(n_channel,len(self.experts), gumbel_tau=gumbel_tau)

        self.mode       = mode # [ 'single', 'moe',  'weight_sharing']

    def _share_weight(self):
        # Copying Direction : 0 -> 1,2,3, ..

        ws_selector = self.experts[0].get_ws_selector()

        for i in range(1,len(self.experts)): 
            for w_key in ws_selector :
                #exec(f'self.experts[i]{w_key} = self.experts[0]{w_key}') #TODO access by getattr and []
                self.experts[i].get_parameter(w_key) = self.experts[0].get_parameter(w_key) ##Why not??
                import pdb; pdb.set_trace()

    def _prepare_weight(self):
        self.experts[0].load_parameter()

        if self.mode == 'ws' :
            # Share Pointer
            self._share_weight()

        elif self.mode == 'moe':
            #Share hard copy of weight
            #for i in range(1,len(self.experts)-1): #0bit
            for i in range(1,len(self.experts)): 
                self.experts[i].load_state_dict(self.experts[0].state_dict())

    def forward(self, x,mask=None):
        if self.mode == 'single' :
            return self.experts[0](x)

        #if self.mode != 'single' and mask == None : 
            #mask = self.gate(x)

        #Random Mask
        #mask = F.one_hot(torch.randn(x.shape[0],len(self.experts)).argmax(dim=1),num_classes=len(self.experts)).float() # Noise-Free
        #mask = mask.to(x.device)

        mask,soft = mask
        self.mask = mask #For Bit loss Computing
        self.soft = soft

        outs = [ expert(x) for expert in self.experts ]
        out = torch.zeros_like(outs[0])
        for i in range(len(outs)):
            out += outs[i]*(mask[:,i].unsqueeze(1).unsqueeze(2).unsqueeze(3))

        return out

class Expert(nn.Module):
    def __init__(self):
        super().__init__()

    def load_parameter(self):
        raise "NOT IMPLEMENTED"

    def get_ws_selector(self):
        # Fully-Qualifed String for torch.nn.Module
        # Eg ) conv1.0.weight
        raise "NOT IMPLEMENTED"

    def forward(self,x):
        raise "NOT IMPLEMENTED"