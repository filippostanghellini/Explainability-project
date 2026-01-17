# src/explainers.py
import torch
import numpy as np
from captum.attr import IntegratedGradients, Saliency, GradientShap, Lime
from captum.attr import visualization as viz

class ExplainerBase:
    """Classe base per tutti gli explainer"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def explain(self, input_tensor, target_class):
        """Deve essere implementato dalle sottoclassi"""
        raise NotImplementedError
    
    def preprocess_attribution(self, attribution):
        """Normalizza attribution map in [0, 1]"""
        attribution = attribution.cpu().detach().numpy()
        
        # Se ha canali RGB, prendi la media
        if len(attribution.shape) == 3:
            attribution = np.mean(np.abs(attribution), axis=0)
        
        # Normalizza
        if attribution.max() > attribution.min():
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
        
        return attribution
    
    def _ensure_int(self, target_class):
        """Converte target_class in int Python"""
        if isinstance(target_class, np.integer):
            return int(target_class)
        elif isinstance(target_class, torch.Tensor):
            return int(target_class.item())
        return int(target_class)


class IntegratedGradientsExplainer(ExplainerBase):
    """Integrated Gradients"""
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.ig = IntegratedGradients(self.model)
    
    def explain(self, input_tensor, target_class=None):
        """
        Args:
            input_tensor: tensor [1, 3, 224, 224]
            target_class: int, classe da spiegare
        
        Returns:
            attribution: numpy array [224, 224]
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Se target_class non specificato, usa la predizione
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Converti a int Python
        target_class = self._ensure_int(target_class)
        
        # Baseline: immagine nera
        baseline = torch.zeros_like(input_tensor).to(self.device)
        
        # Calcola attributions
        attributions = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=50
        )
        
        # Post-process
        attribution_map = self.preprocess_attribution(attributions[0])
        
        return attribution_map, target_class


class InputGradientsExplainer(ExplainerBase):
    """Saliency / Input Gradients"""
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.saliency = Saliency(self.model)
    
    def explain(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Converti a int Python
        target_class = self._ensure_int(target_class)
        
        # Calcola gradients
        attributions = self.saliency.attribute(
            input_tensor,
            target=target_class
        )
        
        attribution_map = self.preprocess_attribution(attributions[0])
        
        return attribution_map, target_class



class IntegratedGradientsExplainer(ExplainerBase):
    """Integrated Gradients"""
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.ig = IntegratedGradients(self.model)
    
    def explain(self, input_tensor, target_class=None):
        """
        Args:
            input_tensor: tensor [1, 3, 224, 224]
            target_class: int, classe da spiegare
        
        Returns:
            attribution: numpy array [224, 224]
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # Se target_class non specificato, usa la predizione
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Converti a int Python
        target_class = self._ensure_int(target_class)
        
        # Baseline: immagine nera
        baseline = torch.zeros_like(input_tensor).to(self.device)
        
        # Calcola attributions con parametri più leggeri
        attributions = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=20,  # <-- RIDOTTO da 50 a 20
            internal_batch_size=5  # <-- AGGIUNTO per processare in batch piccoli
        )
        
        # Post-process
        attribution_map = self.preprocess_attribution(attributions[0])
        
        return attribution_map, target_class


class GradientShapExplainer(ExplainerBase):
    """GradientSHAP (approssimazione di SHAP)"""
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        self.gradient_shap = GradientShap(self.model)
    
    def explain(self, input_tensor, target_class=None, n_samples=50):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Converti a int Python
        target_class = self._ensure_int(target_class)
        
        # Baseline: distribuzione normale attorno a zero
        baselines = torch.randn(n_samples, *input_tensor.shape[1:]).to(self.device) * 0.1
        
        # Calcola attributions
        attributions = self.gradient_shap.attribute(
            input_tensor,
            baselines=baselines,
            target=target_class
        )
        
        attribution_map = self.preprocess_attribution(attributions[0])
        
        return attribution_map, target_class


class LimeExplainer(ExplainerBase):
    """LIME"""
    
    def __init__(self, model, device='cuda'):
        super().__init__(model, device)
        
        # Wrapper per LIME
        def forward_func(input):
            return self.model(input)
        
        self.lime = Lime(forward_func)
    
    def explain(self, input_tensor, target_class=None, n_samples=1000):
        input_tensor = input_tensor.to(self.device)
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = torch.argmax(output, dim=1).item()
        
        # Converti a int Python
        target_class = self._ensure_int(target_class)
        
        # LIME attributions
        attributions = self.lime.attribute(
            input_tensor,
            target=target_class,
            n_samples=n_samples,
            show_progress=False
        )
        
        attribution_map = self.preprocess_attribution(attributions[0])
        
        return attribution_map, target_class