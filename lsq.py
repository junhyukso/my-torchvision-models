from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np 
from collections import OrderedDict


class RoundQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):    
        return input.round()
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n_lv, size):
        ctx.save_for_backward(torch.Tensor([n_lv, size]))
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        saved, = ctx.saved_tensors
        n_lv, size = int(saved[0]), float(saved[1])

        if n_lv == 0:
            return grad_output, None, None
        else:
            scale = 1 / np.sqrt(n_lv * size)
            return grad_output.mul(scale), None, None


class GradientBiReal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        input,  = ctx.saved_tensors
        grad_scale = 2 * torch.abs(input - torch.floor(input + 0.5))
        return input * grad_output


class Q_ReLU(nn.Module):
    def __init__(self,manual_lv=None):
        super(Q_ReLU, self).__init__()
        self.n_lv       = 0
        self.s          = Parameter(torch.Tensor(1))
        self.manual_lv = manual_lv

        #self.sgen = nn.Sequential(
        #       nn.AdaptiveAvgPool2d(1)
        #       nn.Linear(channel,mid_channel,bias=True),
        #       nn.ReLU(inplace=True),
        #       nn.Linear(mid_channel,mid_channel,bias=True),
        #       nn.ReLU(inplace=True),
        #       nn.Linear(mid_channel,1,bias=True),
        #       nn.Sigmoid()
        #        )

    def initialize(self, n_lv, tensor):
        if self.manual_lv != None :
            self.n_lv = self.manual_lv
        else :
            self.n_lv = n_lv
        abs_mean    = tensor.abs().mean()
        self.absmean = abs_mean
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))

        #self.s.data.fill_(1 / np.sqrt(self.n_lv))
    
    def forward(self, x):
        if self.n_lv == 0:
            return x
        else:
            #s = GradientScale.apply(self.s, self.n_lv, x.numel() // x.shape[0])
            s = self.s
            x = F.hardtanh(x / s, 0, self.n_lv - 1)
            x = RoundQuant.apply(x) * s
            return x 

class Q_ReLU_DC(nn.Module):
    def __init__(self,channel,manual_lv=None):
        super(Q_ReLU_DC, self).__init__()
        self.n_lv       = 0
        self.s          = Parameter(torch.Tensor(1))
        self.manual_lv = manual_lv

    def initialize(self, n_lv, tensor):
        if self.manual_lv != None :
            self.n_lv = self.manual_lv
        else :
            self.n_lv = n_lv
        abs_mean    = tensor.abs().mean()
        self.absmean = abs_mean
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))
        #self.s.data.fill_(1 / np.sqrt(self.n_lv))
    
    def forward(self, x, s):
        if self.n_lv == 0:
            return x
        else:
            #s = GradientScale.apply(self.s, self.n_lv, x.numel() // x.shape[0])
            #s = self.sgen(x).view(-1,1,1,1)*2
            s  = s.view(-1,1,1,1)*2
            #se = se.view(-1,1,1,1)

            #x = x/se
            x = F.hardtanh(x / s, 0, self.n_lv - 1)
            x = RoundQuant.apply(x) * s
            return x 

class Q_Sym(nn.Module):
    def __init__(self,init_n_lv = None):
        super(Q_Sym, self).__init__()
        self.n_lv       = 0
        self.s          = Parameter(torch.Tensor(1))
        self.manual_lv = manual_lv

    def initialize(self, n_lv, tensor):
        if self.manual_lv != None :
            self.n_lv = self.manual_lv
        else :
            self.n_lv = n_lv
        abs_mean = tensor.abs().mean()
        self.absmean = abs_mean
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))
        #self.s.data.fill_(1 / np.sqrt(self.n_lv))
    
    def forward(self, x):
        if self.n_lv == 0:
            return x
        else:
            s = GradientScale.apply(self.s, self.n_lv, x.numel() // x.shape[0])
            x = F.hardtanh(x / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            x = RoundQuant.apply(x) * s
            return x


class Q_HSwish(nn.Module):
    def __init__(self):
        super(Q_HSwish, self).__init__()
        self.n_lv = 0
        self.s = Parameter(torch.Tensor(1))

    def initialize(self, n_lv, tensor):
        self.n_lv = n_lv
        abs_mean = tensor.abs().mean()
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))
    
    def forward(self, x):
        if self.n_lv == 0:
            return x
        else:
            s = GradientScale.apply(self.s, self.n_lv, x.numel() // x.shape[0])
            x = F.hardtanh((x + 3/8) / s, 0, self.n_lv - 1)
            x = RoundQuant.apply(x) * s - 3/8
            return x

class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, act_func=None, manual_lv=None, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 0
        self.s = Parameter(torch.Tensor(1))
        self.manual_lv = manual_lv
        

    def initialize(self, n_lv):
        if self.manual_lv != None :
            self.n_lv = self.manual_lv
        else :
            self.n_lv = n_lv
        abs_mean = self.weight.data.abs().mean()
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))


    def _weight_quant(self):
        s = GradientScale.apply(self.s, self.n_lv, self.weight.numel())
        weight = F.hardtanh(self.weight / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        weight = RoundQuant.apply(weight) * s
        return weight
    
    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = RoundQuant.apply(weight)
        return weight

    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)

        if self.n_lv == 0:
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)
        else:
            weight = self._weight_quant()

            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)

class Q_Conv2d_h(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d_h, self).__init__(*args, **kargs)
        self.n_lv = 0
        self.s = Parameter(torch.Tensor(1))

    def initialize(self, n_lv):
        self.n_lv = n_lv
        # abs_mean = F.tanh(self.weight.data).abs().mean()
        abs_mean = self.weight.data.abs().mean()
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))
    
    def _weight_quant(self, weight):
        s = GradientScale.apply(self.s, self.n_lv, self.weight.numel())
        weight_q = F.hardtanh(weight / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        weight_q = RoundQuant.apply(weight_q) * s
        return weight_q

    def forward(self, x):
        if self.n_lv == 0:
            weight = self.weight        
            weight = F.tanh(weight)    
            
        else:
            weight = self.weight        
            weight = F.tanh(weight)

            weight = self._weight_quant(weight)

        return F.conv2d(x, weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups)

"""
class Q_ModConv2d(Q_Conv2d):       
    # TODO: support dilation?
    def forward(self, input):
        if self.n_lv == 0:
            weight = self.weight
        else:
            weight = self._weight_quant()

        default = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)                
        base = torch.ones(input.shape[1], 1, default.shape[2], default.shape[3],  dtype=input.dtype, layout=input.layout, device=input.device, requires_grad=False)        
        
        input_gather = F.conv2d(input, base, None, self.dilation, self.padding, self.stride, groups=input.shape[1])        
        if self.stride[0] == 2:
            input_gather = input_gather[:, :, :-1, :-1]        
        
        offset = F.conv2d(input_gather, weight, None, groups=self.groups)  / (default.shape[2] * default.shape[3])
        return default - offset
"""

class Q_ModConv2d(Q_Conv2d):       
    # TODO: support dilation?
    def forward(self, input):
        if self.n_lv == 0:
            weight = self.weight
        else:
            weight = self._weight_quant()

        default = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)                
        return default - default.mean(dim=(2,3), keepdim=True)


class Q_Conv2dPad(Q_Conv2d):
    def __init__(self, mode, *args, **kargs):
        super(Q_Conv2dPad, self).__init__(*args, **kargs)
        self.mode = mode

    def forward(self, inputs):
        if self.mode == "HS":
            inputs = F.pad(inputs, self.padding + self.padding, value=-0.375)
        elif self.mode == "RE":
            inputs = F.pad(inputs, self.padding + self.padding, value=0)
        else:
            raise LookupError("Unknown nonlinear")

        if self.n_lv == 0:
            return F.conv2d(inputs, self.weight, self.bias,
                self.stride, 0, self.dilation, self.groups)
        else:
            weight = self._weight_quant()

            return F.conv2d(inputs, weight, self.bias,
                self.stride, 0, self.dilation, self.groups)


class Q_Linear(nn.Linear):
    def __init__(self, *args, act_func=None, manual_lv=None, **kargs):
        super(Q_Linear, self).__init__(*args, **kargs)
        self.act_func = act_func
        self.n_lv = 0
        self.s = Parameter(torch.Tensor(1))
        self.manual_lv = manual_lv

    def initialize(self, n_lv):
        if self.manual_lv != None :
            self.n_lv = self.manual_lv
        else :
            self.n_lv = n_lv
        abs_mean = self.weight.data.abs().mean()
        self.s.data.fill_(2 * abs_mean / np.sqrt(self.n_lv))


    def _weight_quant(self):
        s = GradientScale.apply(self.s, self.n_lv, self.weight.numel())
        weight = F.hardtanh(self.weight / s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
        weight = RoundQuant.apply(weight) * s
        return weight

    def _weight_int(self):
        with torch.no_grad():        
            weight = F.hardtanh(self.weight / self.s,- (self.n_lv // 2 - 1), self.n_lv // 2 - 1)
            weight = RoundQuant.apply(weight)
        return weight
        
    def forward(self, x):
        if self.act_func is not None:
            x = self.act_func(x)

        if self.n_lv == 0:
            return F.linear(x, self.weight, self.bias)
        else:
            weight = self._weight_quant()
            return F.linear(x, weight, self.bias)            


def initialize(model, loader, n_lv, act=False, weight=False):

    print("WARNING : you have to call Q.initialize() AFTER .load_state_dict() !")

    def initialize_hook(module, input, output):
        if isinstance(module, (Q_ReLU, Q_ReLU_DC, Q_Sym, Q_HSwish)) and act:
            module.initialize(n_lv, input[0].data)

        if isinstance(module, (Q_Conv2d, Q_Linear)) and weight:
            module.initialize(n_lv)

    hooks = []

    for name, module in model.named_modules():
        hook = module.register_forward_hook(initialize_hook)
        hooks.append(hook)

    
    model.eval()
    #model.train() #SHIT
    model.cpu()
    for i, (input, target) in enumerate(loader):
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                #output = model.module(input.cuda())
                output = model.module(input)
            else:
                #output = model(input.cuda())
                output = model(input)
        break
    
    model.cuda()
    for hook in hooks:
        hook.remove()

# first, last layer 8bit fix
# def initialize(model, loader, n_lv, act=False, weight=False):
#     def initialize_hook(module, input, output):
#         if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:
#             module.initialize(n_lv, input[0].data)

#         if isinstance(module, (Q_Conv2d, Q_Conv2d_h)) and weight:
#                 module.initialize(n_lv)
        
#         if isinstance(module, Q_Linear) and weight: # NOTE Last layer : 8bit
#             module.initialize(2**8)
        
    
#     def initialize_hook_last(module, input, output):
#         if isinstance(module, (Q_ReLU, Q_Sym, Q_HSwish)) and act:  # NOTE Last layer : 8bit
#             module.initialize(2**8, input[0].data)


#     hooks = []


#     for name, module in model.named_modules():
#         if name == 'quant_lin':
#             hook = module.register_forward_hook(initialize_hook_last)
#             hooks.append(hook)
#         else :    
#             hook = module.register_forward_hook(initialize_hook)
#             hooks.append(hook)

    
#     model.train()
#     model.cpu()
#     for i, (input, target) in enumerate(loader):
#         with torch.no_grad():
#             if isinstance(model, nn.DataParallel):
#                 #output = model.module(input.cuda())
#                 output = model.module(input)
#             else:
#                 #output = model(input.cuda())
#                 output = model(input)
#         break
    
#     first=True
#     for n, module in model.named_modules():
#         if isinstance(module, (Q_Conv2d, Q_Conv2d_h)) and weight:
#             if first :
#                 module.initialize(2**8) # NOTE first conv : 8bit
#                 first=False
#                 break

#     model.cuda()
#     for hook in hooks:
#         hook.remove()


class Q_Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Q_Sequential, self).__init__()

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0 
            for module in args:
                if isinstance(module, (Q_Sym, Q_ReLU)) or (isinstance(module, Q_HSwish) and idx == 0):
                    self.add_module('-' + str(idx), module)
                else:
                    self.add_module(str(idx), module)
                    idx += 1


class QuantOps(object):
    initialize = initialize
    ReLU = Q_ReLU
    ReLU_DC = Q_ReLU_DC
    ReLU6 = Q_ReLU
    Sym = Q_Sym
    HSwish = Q_HSwish
    Conv2d = Q_Conv2d
    ModConv2d = Q_ModConv2d
    Conv2dPad = Q_Conv2dPad
    Linear = Q_Linear
    Sequential = Q_Sequential
    
