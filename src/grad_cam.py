import torch
import numpy as np
import cv2


class SimpleGradCAM:
    """Gradient-weighted Class Activation Mapping for CNN interpretability."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_heatmap(self, input_tensor):
        """Returns (heatmap_array, predicted_class_index)."""
        output = self.model(input_tensor)
        idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        output[0, idx].backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap /= (torch.max(heatmap) + 1e-10)
        return heatmap.detach().cpu().numpy(), idx

    def overlay(self, heatmap, original_bgr, alpha=0.6):
        """Superimposes the heatmap on the original BGR image. Returns RGB array."""
        h = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
        h_color = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
        h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
        rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(rgb, alpha, h_color, 1 - alpha, 0)
