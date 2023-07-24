from captum.attr import LayerGradientXActivation, GradientShap, DeepLift, Saliency
from captum.attr._utils.gradient import apply_gradient_requirements, undo_gradient_requirements

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.gradient_method = Saliency(self.model)

    def attribute(self, input_data, target=0, abs_sum=False):
        self.gradient_method.gradient_func = self._gradient_override

        self._change_hooks()
        self.activations = []
        self.gradients = []
        self.backward_hooks = []
        self.forward_hooks = []

        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.forward_hooks.append(module.register_forward_hook(self._forward_hook))
                self.backward_hooks.append(module.register_backward_hook(self._backward_hook))

        attributions = self.gradient_method.attribute(input_data, target=target)

        if abs_sum:
            attributions = attributions.abs().sum(dim=1)

        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        return attributions

    def _gradient_override(self, grad):
        return grad / (self.activations[-1] + 1e-10)

    def _change_hooks(self):
        self.model._modules['fc']._backward_hooks.clear()
        self.model._modules['fc'].register_backward_hook(self._hook_fc)

    def _forward_hook(self, module, input, output):
        self.activations.append(output)

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0] * self.activations[-1])

    def _hook_fc(self, module, grad_in, grad_out):
        return (self.activations[-1].mean(dim=-1, keepdim=True) * grad_in[0],)
