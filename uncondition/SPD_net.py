import torch
from torch import nn
from torch.autograd import Function
import numpy as np

from support_function import *
from math import ceil

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


class SPD_NET(nn.Module):
    def __init__(self, spd_size,time_size):
        super().__init__()
        self.time_dim = time_size
        spd_size_div2 = spd_size//2
        spd_size_div4 = spd_size_div2//2
        
        self.trans1 = SPDTransform(spd_size, spd_size, self.time_dim)
        self.trans1_5 = SPDTransform(spd_size, spd_size, self.time_dim)

        self.trans2 = SPDTransform(spd_size, spd_size, self.time_dim)
        self.trans2_5 = SPDTransform(spd_size, spd_size, self.time_dim)

        self.trans3 = SPDTransform(spd_size, spd_size_div2, self.time_dim)
        self.trans3_5= SPDTransform(spd_size_div2, spd_size_div2, self.time_dim)

        self.trans4 = SPDTransform(spd_size_div2, spd_size_div2, self.time_dim)
        self.trans4_5 = SPDTransform(spd_size_div2, spd_size_div2, self.time_dim)

        self.trans5 = SPDTransform(spd_size_div2, spd_size_div4, self.time_dim)
        self.trans5_5 = SPDTransform(spd_size_div4, spd_size_div4, self.time_dim) 

        self.trans6 = SPDTransform(spd_size_div4, spd_size_div4, self.time_dim)
        self.trans6_5 = SPDTransform(spd_size_div4, spd_size_div4, self.time_dim)

        self.trans7 = SPDTransform(spd_size_div4, spd_size_div2, self.time_dim)
        self.trans7_5 = SPDTransform(spd_size_div2, spd_size_div2, self.time_dim)

        self.trans8 = SPDTransform(spd_size_div2, spd_size_div2, self.time_dim)
        self.trans8_5 = SPDTransform(spd_size_div2, spd_size_div2, self.time_dim)

        self.trans9 = SPDTransform(spd_size_div2, spd_size, self.time_dim)
        self.trans9_5 = SPDTransform(spd_size, spd_size, self.time_dim)
    
        self.trans10 = SPDTransform(spd_size, spd_size, self.time_dim)
        self.trans10_5= SPDTransform(spd_size, spd_size, self.time_dim)

        self.trans11 = SPDTransform(spd_size, spd_size, self.time_dim)

        self.rect1  = SPDRectified()
        self.rect1_5  = SPDRectified()
        self.rect2  = SPDRectified()
        self.rect2_5  = SPDRectified()
        self.rect3  = SPDRectified()
        self.rect3_5  = SPDRectified()
        self.rect4  = SPDRectified()
        self.rect4_5  = SPDRectified()
        self.rect5  = SPDRectified()
        self.rect5_5  = SPDRectified()
        self.rect5_8  = SPDRectified()
        self.rect5_9  = SPDRectified()
        self.rect6 = SPDRectified()
        self.rect6_5 = SPDRectified()
        self.rect7 = SPDRectified()
        self.rect7_5 = SPDRectified()
        self.rect8 = SPDRectified()
        self.rect8_5 = SPDRectified()
        self.rect9 = SPDRectified()
        self.rect9_5 = SPDRectified() 
        self.rect10 = SPDRectified()
        self.rect10_5 = SPDRectified()
        self.rect11 = SPDRectified()  

    def pos_encoding(self, t, channels):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inv_freq = 1.0 / (10000** (torch.arange(0, channels, 2, device=device)/ channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2 )* inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2 ) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1).double()
        return pos_enc

    def unet_forwad(self, x, t):
        
        r = 0.5
        x1 = self.trans1(x,t)   
        x1 = self.rect1(x1)

        x1_5 = self.trans1_5(x1,t)   
        x1_5= self.rect1_5(x1_5)

        x2 = self.trans2(x1_5,t)  
        x2 = self.rect2(x2)
        x2_5 = self.trans2_5(x2,t)  
        x2_5 = self.rect2_5(x2_5)

        x3 = self.trans3(x2_5,t)  
        x3 = self.rect3(x3)
        x3_5 = self.trans3_5(x3,t)  
        x3_5 = self.rect3_5(x3_5)

        x4 = self.trans4(x3_5,t)  
        x4 = self.rect4(x4)
        x4_5 = self.trans4_5(x4,t)  
        x4_5 = self.rect4_5(x4_5)

        x5 = self.trans5(x4_5,t)  
        x5 = self.rect5(x5)
        x5_5 = self.trans5_5(x5,t)  
        x5_5 = self.rect5_5(x5_5)

        x6 = self.trans6(x5_5,t)  
        x6 = self.rect6(x6)
        x6_5 = self.trans6_5(x6,t)  
        x6_5 = self.rect6_5(x6_5)

        x7 = (r*x6_5 + (1-r)*x5_5)/2  
        x7 = self.trans7(x7,t)  
        x7 = self.rect7(x7)
        x7_5 = self.trans7_5(x7,t)  
        x7_5 = self.rect7_5(x7_5)

        x8= (r*x7_5 + (1-r)*x4_5)/2
        x8 = self.trans8(x8,t)  
        x8 = self.rect8(x8)
        x8_5 = self.trans8_5(x8,t)  
        x8_5 = self.rect8_5(x8_5)

        x9 = (r*x8_5 + (1-r)*x3_5)/2
        x9 = self.trans9(x9,t)  
        x9 = self.rect9(x9)
        x9_5 = self.trans9_5(x9,t)  
        x9_5 = self.rect9_5(x9_5)

        x10 = (r*x9_5 + (1-r)*x2_5)/2
        x10 = self.trans10(x10,t)  
        x10 = self.rect10(x10)
        x10_5 = self.trans10_5(x10,t)  
        x10_5 = self.rect10_5(x10_5)

        x = self.trans11(x10_5,t)  
        
        return x
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        result = self.unet_forwad(x, t)
        return result


class SPDTransform(nn.Module):

    def __init__(self, input_size , output_size, time_dim):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        input_size = int(input_size)
        output_size = int(output_size)
        time_dim = int(time_dim)
        self.input_size = input_size

        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        self.weight = StiefelParameter(torch.DoubleTensor(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

        self.emb_layer = nn.Sequential( 
            nn.Linear(time_dim,time_dim).double(),
            nn.SiLU(), 
            nn.Linear(time_dim,output_size*output_size).double(),
        )

    def forward(self, input,t):
        output = input
        emb = self.emb_layer(t)   
        

        if self.increase_dim:
            output = self.increase_dim(output)

        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight.transpose(1,2), torch.bmm(output, weight))
        
        m = output.shape[1]
        n = output.shape[0]

        t_emb = emb.reshape(n,m,m)

        output = torch.matmul(torch.matmul(t_emb,output),t_emb.transpose(1,2))
        
        
        return output

class StiefelParameter(nn.Parameter):
    """A kind of Variable that is to be considered a module parameter on the space of 
        Stiefel manifold.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
    
class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        #epsilon = -0.01
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output


class SPDRectifiedFunction(Function):

    @staticmethod
    def forward(ctx, input, epsilon):
        ctx.save_for_backward(input, epsilon)

        U, S, Vh = torch.linalg.svd(input)
        S = torch.clamp(S, min=epsilon[0])
        S_diag = torch.diag_embed(S)
        output = U.bmm(S_diag.bmm(U.transpose(-1, -2)))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, epsilon = ctx.saved_variables
        grad_input = None
        
        if ctx.needs_input_grad[0]:
            U, S, Vh = torch.linalg.svd(input)
            UT_g = torch.bmm(U.transpose(-2, -1), grad_output)
            UT_g_u = torch.bmm(UT_g, U)
            max_mask = S > epsilon 
            Q = max_mask
            S_max_diag = torch.clamp(S, min=epsilon[0]) 
            S_max_diag_mat = torch.diag_embed(S_max_diag)
            dLdS_diag = torch.diag_embed(U.new_tensor(Q) * torch.diagonal(UT_g_u, dim1=-2, dim2=-1)) 
            dLdV = 2 * torch.bmm(grad_output, torch.bmm(U, S_max_diag_mat)) 
            
            s_expand_row = S.unsqueeze(-1) 
            s_expand_col = S.unsqueeze(-2) 
            P = s_expand_row - s_expand_col 
            
            mask_zero = torch.abs(P) < 1e-9
            P_inv = torch.where(mask_zero, torch.zeros_like(P), 1.0 / P)
            
            UT_dLdV = torch.bmm(U.transpose(-2, -1), dLdV) 
            A = P_inv.transpose(-2, -1) * UT_dLdV 
            
            sym_term = 0.5 * (A + A.transpose(-2, -1))
            grad_input = torch.bmm(U, torch.bmm(sym_term + dLdS_diag, U.transpose(-2, -1)))
            
        return grad_input, None


def symmetric(A):
    return 0.5 * (A + A.t())


class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = np.asarray([0] * input_size + [1] * (output_size-input_size), dtype=np.float64)
        self.register_buffer('add', torch.from_numpy(np.diag(add)))

    def forward(self, input):
        eye = self.eye.unsqueeze(0)
        eye = eye.expand(input.size(0), -1, -1).double()
        add = self.add.unsqueeze(0)
        add = add.expand(input.size(0), -1, -1)

        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1,2)))  

        return output

model = SPD_NET(8,100)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")