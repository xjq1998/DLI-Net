import torch
from torch.nn import ReLU


def cal_euclidean_dist(ph_fea, sk_fea):
    dist = torch.pow(ph_fea-sk_fea, 2)
    dist = torch.sum(dist)
    dist = torch.sqrt(dist)
    return dist


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, ph_encoder, sk_encoder, self_interaction, match):
        self.ph_encoder = ph_encoder
        self.sk_encoder = sk_encoder
        self.self_interaction=self_interaction
        self.match = match
        self.ph_gradients = None
        self.sk_gradients = None
        self.ph_forward_relu_outputs = []
        self.sk_forward_relu_outputs = []
        # Put model in evaluation mode
        self.ph_encoder.eval()
        self.sk_encoder.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_ph_function(module, grad_in, grad_out):
            self.ph_gradients = grad_in[0]

        def hook_sk_function(module, grad_in, grad_out):
            self.sk_gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.ph_encoder.backbone._modules.items())[0][1]
        first_layer.register_backward_hook(hook_ph_function)
        first_layer = list(self.sk_encoder.backbone._modules.items())[0][1]
        first_layer.register_backward_hook(hook_sk_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def ph_relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.ph_forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * \
                torch.clamp(grad_in[0], min=0.0)
            del self.ph_forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)
        
        def sk_relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.sk_forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * \
                torch.clamp(grad_in[0], min=0.0)
            del self.sk_forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def ph_relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.ph_forward_relu_outputs.append(ten_out)
        
        def sk_relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.sk_forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.ph_encoder.backbone._modules.items():
            if isinstance(module, ReLU):
                module.register_forward_hook(ph_relu_forward_hook_function)
                module.register_backward_hook(ph_relu_backward_hook_function)
        for pos, module in self.sk_encoder.backbone._modules.items():
            if isinstance(module, ReLU):
                module.register_forward_hook(sk_relu_forward_hook_function)
                module.register_backward_hook(sk_relu_backward_hook_function)
                

    def generate_gradients(self, ph, sk, index):
        # Forward
        ph_fea = self.ph_encoder(ph)
        sk_fea = self.sk_encoder(sk)
        ph_fea=self.self_interaction(ph_fea)
        sk_fea=self.self_interaction(sk_fea)
        # Zero grads
        self.ph_encoder.zero_grad()
        self.sk_encoder.zero_grad()
        # Loss for backprop
        local_dis = self.match.get_local_dis(ph_fea, sk_fea,index)[0][0]
        # Backward pass
        local_dis.backward()
        # Convert Pytorch variable to numpy array
        gradients_ph = self.ph_gradients.data.cpu().numpy()[0]
        gradients_sk = self.sk_gradients.data.cpu().numpy()[0]
        return gradients_ph, gradients_sk
