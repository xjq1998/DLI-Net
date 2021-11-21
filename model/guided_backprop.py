import torch
from torch.nn import ReLU


def cal_euclidean_dist(img_fea, sk_fea):
    dist = torch.pow(img_fea-sk_fea, 2)
    dist = torch.sum(dist)
    dist = torch.sqrt(dist)
    return dist


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, img_encoder, sk_encoder, self_interaction, match):
        self.img_encoder = img_encoder
        self.sk_encoder = sk_encoder
        self.self_interaction=self_interaction
        self.match = match
        self.img_gradients = None
        self.sk_gradients = None
        self.img_forward_relu_outputs = []
        self.sk_forward_relu_outputs = []
        # Put model in evaluation mode
        self.img_encoder.eval()
        self.sk_encoder.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_img_function(module, grad_in, grad_out):
            self.img_gradients = grad_in[0]

        def hook_sk_function(module, grad_in, grad_out):
            self.sk_gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.img_encoder.backbone._modules.items())[0][1]
        first_layer.register_backward_hook(hook_img_function)
        first_layer = list(self.sk_encoder.backbone._modules.items())[0][1]
        first_layer.register_backward_hook(hook_sk_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def img_relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.img_forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * \
                torch.clamp(grad_in[0], min=0.0)
            del self.img_forward_relu_outputs[-1]  # Remove last forward output
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

        def img_relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.img_forward_relu_outputs.append(ten_out)
        
        def sk_relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.sk_forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.img_encoder.backbone._modules.items():
            if isinstance(module, ReLU):
                module.register_forward_hook(img_relu_forward_hook_function)
                module.register_backward_hook(img_relu_backward_hook_function)
        for pos, module in self.sk_encoder.backbone._modules.items():
            if isinstance(module, ReLU):
                module.register_forward_hook(sk_relu_forward_hook_function)
                module.register_backward_hook(sk_relu_backward_hook_function)
                

    def generate_gradients(self, img, sk):
        # Forward
        img_fea = self.img_encoder(img)
        sk_fea = self.sk_encoder(sk)
        img_fea=self.self_interaction(img_fea)
        sk_fea=self.self_interaction(sk_fea)
        # Zero grads
        self.img_encoder.zero_grad()
        self.sk_encoder.zero_grad()
        # Loss for backprop
        dis = self.match(img_fea, sk_fea)
        # loss = self.match.cal_local_dis(img_fea, sk_fea)[186,0,0]
        # loss=cal_euclidean_dist(img_fea,sk_fea)
        # Backward pass
        dis.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_img = self.img_gradients.data.cpu().numpy()[0]
        gradients_sk = self.sk_gradients.data.cpu().numpy()[0]
        return gradients_img, gradients_sk
