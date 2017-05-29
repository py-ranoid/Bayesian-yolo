import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Function
import bn_lib

class BN2dFunc(Function):
    def __init__(self, running_mean, running_var, training, momentum, eps):
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, weight, bias):
        nB = input.size(0)
        nC = input.size(1)
        nH = input.size(2)
        nW = input.size(3)

        output = input.new(nB, nC, nH, nW) 
        self.input = input
        self.weight = weight
        self.bias = bias
        self.x = input.new(nB, nC, nH, nW) 
        self.x_norm = input.new(nB, nC, nH, nW) 
        self.mean = input.new(nB, nC) 
        self.var = input.new(nB, nC) 
        print 'type:', type(self.mean)

        if input.is_cuda:
            bn_lib.bn_forward_gpu(input, self.x, self.x_norm, self.mean, self.running_mean, self.var, self.running_var, weight, bias, self.training, output)
        else:
            print 'using cpu'
            bn_lib.bn_forward(input, self.x, self.x_norm, self.mean, self.running_mean, self.var, self.running_var, weight, bias, self.training, output)
        return output

    def backward(self, grad_output):
        nB = grad_output.size(0)
        nC = grad_output.size(1)
        nH = grad_output.size(2)
        nW = grad_output.size(3)
        grad_input = grad_output.new(nB, nC, nH, nW) 
        grad_mean = grad_output.new(nC) 
        grad_var = grad_output.new(nC) 
        grad_weight = grad_output.new(nC) 
        grad_bias = grad_output.new(nC) 
        
        if grad_output.is_cuda:
            bn_lib.bn_backward_gpu(grad_output, self.input, self.x_norm, self.mean, grad_mean, self.var, grad_var, self.weight, grad_weight, self.bias, grad_bias, self.training, grad_input)
        else:
            print 'using cpu'
            bn_lib.bn_backward(grad_output, self.input, self.x_norm, self.mean, grad_mean, self.var, grad_var, self.weight, grad_weight, self.bias, grad_bias, self.training, grad_input)
        
        return grad_input, grad_weight, grad_bias  

class BN2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-5):
        super(BN2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.eps = eps

        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, input):
        return BN2dFunc(self.running_mean, self.running_var, self.training, self.momentum, self.eps)(input, self.weight, self.bias)

if __name__ == '__main__':
    a = torch.rand(3,3,2,2).cuda()
    #a = torch.rand(3,3,2,2)
    a = Variable(a)
    m1 = nn.BatchNorm2d(3)
    m2 = BN2d(3)
    m1.cuda()
    m2.cuda()
    #m1.eval()
    #m2.eval()
    m1.weight.data.fill_(1)
    m1.bias.data.zero_()
    m2.weight.data.fill_(1)
    m2.bias.data.zero_()
    b = m1(a)
    c = m2(a)
    print(b)
    print(c)

