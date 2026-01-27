"""
Explainability methods using Captum library.
Implements LIME, SHAP (KernelSHAP), Integrated Gradients, and Input Gradients.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    Saliency,
    Lime,
    KernelShap,
    GradientShap,
    Occlusion,
    NoiseTunnel
)
from skimage.segmentation import slic

from . import config


class ExplainabilityMethods:
    """
    Wrapper class for various explainability methods from Captum.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize explainability methods.
        
        Args:
            model: Trained PyTorch model
            device: Device to run computations on
        """
        self.model = model
        self.model.eval()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Initialize attribution methods
        self._init_methods()
    
    def _init_methods(self):
        """Initialize all attribution methods."""
        # Gradient-based methods
        self.integrated_gradients = IntegratedGradients(self.model)
        self.input_x_gradient = InputXGradient(self.model)
        self.saliency = Saliency(self.model)
        
        # Perturbation-based methods
        self.occlusion = Occlusion(self.model)
        
        # LIME and SHAP
        self.lime = Lime(self.model)
        self.kernel_shap = KernelShap(self.model)
        self.gradient_shap = GradientShap(self.model)
    
    def get_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_steps: int = 25,
        baselines: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.
        
        Integrated Gradients accumulates gradients along a path from a baseline
        (typically a black image) to the input image.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            n_steps: Number of steps for integration
            baselines: Baseline tensor (default: zeros)
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if baselines is None:
            baselines = torch.zeros_like(input_tensor).to(self.device)
        
        attributions = self.integrated_gradients.attribute(
            input_tensor,
            baselines=baselines,
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        # Convert to grayscale attribution map
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
    def get_input_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Compute Input × Gradient attribution.
        
        This method multiplies the input by the gradient of the output
        with respect to the input (element-wise).
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        attributions = self.input_x_gradient.attribute(
            input_tensor,
            target=target_class
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    

    def get_saliency(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        abs_value: bool = True
    ) -> np.ndarray:
        """
        Compute Saliency (Vanilla Gradients) attribution.
        
        Simple gradient of the output with respect to input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            abs_value: Whether to take absolute value
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        attributions = self.saliency.attribute(
            input_tensor,
            target=target_class,
            abs=abs_value
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
    def get_lime(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_samples: int = 1000,
        feature_mask: Optional[torch.Tensor] = None,
        n_segments: int = 50
    ) -> np.ndarray:
        """
        Compute LIME (Local Interpretable Model-agnostic Explanations) attribution.
        
        LIME creates local linear approximations around the input by perturbing
        superpixels (segments) of the image.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            n_samples: Number of perturbation samples
            feature_mask: Optional segmentation mask
            n_segments: Number of segments if feature_mask not provided
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        
        # Create feature mask using SLIC if not provided
        if feature_mask is None:
            feature_mask = self._create_segmentation_mask(
                input_tensor,  # Passa l'immagine intera!
                n_segments
            )
        
        # Expand feature mask to match input channels
        if feature_mask.dim() == 2:
            feature_mask = feature_mask.unsqueeze(0).unsqueeze(0)
            feature_mask = feature_mask.expand(1, input_tensor.shape[1], -1, -1)
        
        attributions = self.lime.attribute(
            input_tensor,
            target=target_class,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=min(16, n_samples), #TODO: test with 125 or 256 using GPU
            show_progress=False
        )
        
        # Perturbation-based: preserve sign for divergent colormap
        attr_map = self._to_grayscale(attributions, take_abs=False)
        return attr_map
    
    def get_kernel_shap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_samples: int = 100,
        feature_mask: Optional[torch.Tensor] = None,
        n_segments: int = 50,
        baselines: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute Kernel SHAP attribution.
        
        SHAP (SHapley Additive exPlanations) computes Shapley values
        using a weighted linear regression approach.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            n_samples: Number of perturbation samples
            feature_mask: Optional segmentation mask
            n_segments: Number of segments if feature_mask not provided
            baselines: Baseline tensor for SHAP
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        
        if baselines is None:
            baselines = torch.zeros_like(input_tensor).to(self.device)
        
        # Create feature mask using SLIC if not provided
        if feature_mask is None:
            feature_mask = self._create_segmentation_mask(
                input_tensor,  # Passa l'immagine intera!
                n_segments
            )
        
        # Expand feature mask to match input channels
        if feature_mask.dim() == 2:
            feature_mask = feature_mask.unsqueeze(0).unsqueeze(0)
            feature_mask = feature_mask.expand(1, input_tensor.shape[1], -1, -1)
        
        attributions = self.kernel_shap.attribute(
            input_tensor,
            baselines=baselines,
            target=target_class,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=min(16, n_samples), #TODO: test with 125 or 256 using GPU
            show_progress=False
        )
        
        # Perturbation-based: preserve sign for divergent colormap
        attr_map = self._to_grayscale(attributions, take_abs=False)
        return attr_map

    def get_occlusion(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        sliding_window_shapes: Tuple[int, int, int] = (3, 15, 15),
        strides: Tuple[int, int, int] = (3, 8, 8)
    ) -> np.ndarray:
        """
        Compute Occlusion attribution.
        
        Systematically occlude parts of the input and measure 
        the change in output.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            sliding_window_shapes: Shape of the occlusion patch
            strides: Stride for sliding the window
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        
        attributions = self.occlusion.attribute(
            input_tensor,
            target=target_class,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=0
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
    def get_gradient_shap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_samples: int = 50,
        baselines: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute Gradient SHAP attribution.
        
        Gradient SHAP combines SHAP with gradient information for efficiency.
        More efficient than Kernel SHAP as it uses gradients instead of perturbation.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            n_samples: Number of samples for baseline averaging
            baselines: Baseline tensor for SHAP (default: random noise)
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        
        if baselines is None:
            # Generate random baselines
            baselines = torch.randn_like(input_tensor).to(self.device)
        
        attributions = self.gradient_shap.attribute(
            input_tensor,
            baselines=baselines,
            target=target_class,
            n_samples=n_samples,
            stdevs=0.09
        )
        
        # Gradient-based SHAP: use absolute value for hot colormap visualization
        attr_map = self._to_grayscale(attributions, take_abs=True)
        return attr_map
    
    def get_integrated_gradients_with_noise(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_steps: int = 25,
        baselines: Optional[torch.Tensor] = None,
        nt_type: str = 'smoothgrad',
        nt_samples: int = 10,   
        stdevs: float = 0.15
    ) -> np.ndarray:
        """
        Compute Integrated Gradients with NoiseTunnel (SmoothGrad variant).
        
        Adds noise to the input multiple times, computes attributions for each,
        and averages them for more robust explanations.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            n_steps: Number of steps for integration
            baselines: Baseline tensor (default: zeros)
            nt_type: Noise tunnel type ('smoothgrad' or 'vargrad')
            nt_samples: Number of noise samples to average
            stdevs: Standard deviation of noise
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        if baselines is None:
            baselines = torch.zeros_like(input_tensor).to(self.device)
        
        # Wrap Integrated Gradients with NoiseTunnel
        ig_with_noise = NoiseTunnel(self.integrated_gradients)
        
        attributions = ig_with_noise.attribute(
            input_tensor,
            baselines=baselines,
            target=target_class,
            n_steps=n_steps,
            nt_type=nt_type,
            nt_samples=nt_samples,
            stdevs=stdevs,
            return_convergence_delta=False
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
    def get_saliency_with_noise(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        abs_value: bool = True,
        nt_type: str = 'smoothgrad',
        nt_samples: int = 10,
        stdevs: float = 0.15
    ) -> np.ndarray:
        """
        Compute Saliency with NoiseTunnel for improved robustness.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            abs_value: Whether to take absolute value
            nt_type: Noise tunnel type ('smoothgrad' or 'vargrad')
            nt_samples: Number of noise samples to average
            stdevs: Standard deviation of noise
            
        Returns:
            Attribution map as numpy array (H, W)
        """
        assert input_tensor.shape[0] == 1, "Attribution methods work on single images only (batch_size=1)"
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        saliency_with_noise = NoiseTunnel(self.saliency)
        
        attributions = saliency_with_noise.attribute(
            input_tensor,
            target=target_class,
            abs=abs_value,
            nt_type=nt_type,
            nt_samples=nt_samples,
            stdevs=stdevs
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
    def get_all_attributions(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        methods: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute attributions using multiple methods.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class for attribution
            methods: List of method names to use
            
        Returns:
            Dictionary mapping method names to attribution maps
        """
        if methods is None:
            methods = config.EXPLAINABILITY_METHODS
        
        attributions = {}
        
        for method in methods:
            try:
                if method == 'integrated_gradients':
                    attributions[method] = self.get_integrated_gradients(
                        input_tensor, target_class
                    )
                elif method == 'input_gradients':
                    attributions[method] = self.get_input_gradients(
                        input_tensor, target_class
                    )
                elif method == 'saliency':
                    attributions[method] = self.get_saliency(
                        input_tensor, target_class
                    )
                elif method == 'lime':
                    attributions[method] = self.get_lime(
                        input_tensor, target_class,
                        n_samples=config.N_SAMPLES_LIME
                    )
                elif method == 'kernel_shap':
                    attributions[method] = self.get_kernel_shap(
                        input_tensor, target_class,
                        n_samples=config.N_SAMPLES_SHAP
                    )
                elif method == 'gradient_shap':
                    attributions[method] = self.get_gradient_shap(
                        input_tensor, target_class,
                        n_samples=config.N_SAMPLES_SHAP
                    )
                elif method == 'occlusion':
                    attributions[method] = self.get_occlusion(
                        input_tensor, target_class
                    )
                elif method == 'integrated_gradients_noise':
                    attributions[method] = self.get_integrated_gradients_with_noise(
                        input_tensor, target_class,
                        nt_samples=config.NT_SAMPLES if hasattr(config, 'NT_SAMPLES') else 10
                    )
                elif method == 'saliency_noise':
                    attributions[method] = self.get_saliency_with_noise(
                        input_tensor, target_class,
                        nt_samples=config.NT_SAMPLES if hasattr(config, 'NT_SAMPLES') else 10
                    )
            except Exception as e:
                print(f"Error computing {method}: {e}")
                attributions[method] = None
        
        return attributions
    
    def _to_grayscale(self, attributions: torch.Tensor, take_abs: bool = True) -> np.ndarray:
        """
        Convert attribution tensor to grayscale numpy array.
        
        Args:
            attributions: Attribution tensor from Captum
            take_abs: If True, take absolute value (for gradient-based methods).
                      If False, preserve sign (for perturbation-based methods).
        
        Returns:
            Attribution map as numpy array (H, W)
        """
        # Sum across channels
        attr_np = attributions.squeeze().cpu().detach().numpy()
        
        if attr_np.ndim == 3:  # (C, H, W)
            if take_abs:
                attr_np = np.abs(attr_np).sum(axis=0)
            else:
                # Per metodi perturbation-based: preserva il segno
                attr_np = attr_np.sum(axis=0)
        else:  # Already (H, W)
            if take_abs:
                attr_np = np.abs(attr_np)
            # else: mantieni attr_np così com'è (con segno)
        
        return attr_np

    #TODO: original function
    # def _create_segmentation_mask(self, input_tensor: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Create a segmentation mask using SLIC superpixels.
        
        Uses SLIC (Simple Linear Iterative Clustering) instead of a simple grid
        for more semantically meaningful superpixel segmentation.
        """
        # Porta su CPU e converti in numpy
        img_np = input_tensor.cpu().detach().numpy()
        
        # Gestione dimensioni: se c'è la dimensione batch (1, C, H, W), rimuovila
        if img_np.ndim == 4:
            img_np = img_np[0]  # Diventa (C, H, W)
            
        # Trasponi da (C, H, W) a (H, W, C) per scikit-image
        # Controlliamo se la prima dimensione è i canali (3)
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
            
        # Calcola segmenti con SLIC
        # start_label=0 è fondamentale per far funzionare feature_mask con Captum
        segments = slic(img_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
        
        # Ritorna come tensore Long (intero) sul device corretto
        return torch.from_numpy(segments).long().to(self.device)

    def _create_segmentation_mask(self, input_tensor: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Create a segmentation mask using SLIC superpixels.
        
        Uses SLIC (Simple Linear Iterative Clustering) instead of a simple grid
        for more semantically meaningful superpixel segmentation.
        """
        # Porta su CPU e converti in numpy
        img_np = input_tensor.cpu().detach().numpy()
        
        # Gestione dimensioni: se c'è la dimensione batch (1, C, H, W), rimuovila
        if img_np.ndim == 4:
            img_np = img_np[0]  # Diventa (C, H, W)
            
        # Trasponi da (C, H, W) a (H, W, C) per scikit-image
        if img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # DE-NORMALIZZA per SLIC: riporta i valori in [0, 1]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_denorm = img_np * std + mean
        img_denorm = np.clip(img_denorm, 0, 1)  # Assicura range valido
            
        # Calcola segmenti con SLIC su immagine de-normalizzata
        segments = slic(img_denorm, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
        
        return torch.from_numpy(segments).long().to(self.device)