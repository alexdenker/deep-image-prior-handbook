


import torch 

class OperatorModule(torch.nn.Module):
    def __init__(self, operator):
        """Initialize a new instance."""
        super(OperatorModule, self).__init__()
        self.operator = operator

    def forward(self, x):
        return OperatorFunction.apply(self.operator, x)
    
    def A_dagger(self, y):
        return self.operator.A_dagger(y)

    def A_adjoint(self, y):
        return self.operator.A_adjoint(y)

    def A(self, y):
        return self.operator.A(y)

class OperatorFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, operator, input):
        
        # Save operator for backward; input only needs to be saved if
        # the operator is nonlinear (for `operator.derivative(input)`)
        ctx.operator = operator

        out = operator.A(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        operator = ctx.operator

        grad_input = operator.A_adjoint(grad_output)

        return None, grad_input  # return `None` for the `operator` part