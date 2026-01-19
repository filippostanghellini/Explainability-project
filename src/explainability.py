"""
Explainability methods using Captum library.
Implements LIME, SHAP (KernelSHAP), Integrated Gradients, and Input Gradients.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
from captum.attr import (
    IntegratedGradients,
    InputXGradient,
    Saliency,
    GuidedBackprop,
    GuidedGradCam,
    Lime,
    KernelShap,
    Occlusion,
    NoiseTunnel, #TODO: non viene utilizzato, "wrapper" che prende un metodo (es. Integrated Gradients), aggiunge rumore all'immagine più volte, calcola le spiegazioni e ne fa la media.
    LayerGradCam,
    LayerAttribution
)
from captum.attr._utils.visualization import visualize_image_attr
from captum._utils.models.linear_model import SkLearnRidge
from skimage.segmentation import slic
import matplotlib.pyplot as plt

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
        self.guided_backprop = GuidedBackprop(self.model)
        
        # Perturbation-based methods
        self.occlusion = Occlusion(self.model)
        
        # LIME and SHAP need special handling
        self.lime = Lime(self.model)
        self.kernel_shap = KernelShap(self.model)
    
    def get_integrated_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_steps: int = 50,
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
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        attributions = self.input_x_gradient.attribute(
            input_tensor,
            target=target_class
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
#TODO: comprendere se va rimosso dato che non è richiesto

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
        n_segments: int = 100 #INFO: parametro per SLIC (50 non da problemi)
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
            perturbations_per_eval=min(16, n_samples),
            show_progress=False
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map
    
    def get_kernel_shap(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_samples: int = 100,
        feature_mask: Optional[torch.Tensor] = None,
        n_segments: int = 100, #INFO: parametro per SLIC (50 non da problemi)
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
            perturbations_per_eval=min(16, n_samples),
            show_progress=False
        )
        
        attr_map = self._to_grayscale(attributions)
        return attr_map

#TODO: comprendere se va rimosso dato che non è richiesto

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
                elif method == 'occlusion':
                    attributions[method] = self.get_occlusion(
                        input_tensor, target_class
                    )
            except Exception as e:
                print(f"Error computing {method}: {e}")
                attributions[method] = None
        
        return attributions
    
    def _to_grayscale(self, attributions: torch.Tensor) -> np.ndarray:
        """
        Convert attribution tensor to grayscale numpy array.
        
        Takes the sum of absolute values across channels.
        """
        # Sum across channels and take absolute value
        attr_np = attributions.squeeze().cpu().detach().numpy()
        
        if attr_np.ndim == 3:  # (C, H, W)
            attr_np = np.abs(attr_np).sum(axis=0)
        else:  # Already (H, W)
            attr_np = np.abs(attr_np)
        
        return attr_np

#INFO: utilizziamo SLIC e non un semplice grid per segmentare l'immagine in superpixels

    def _create_segmentation_mask(self, input_tensor: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Create a segmentation mask using SLIC superpixels.
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
    
    def normalize_attribution(self, attr_map: np.ndarray) -> np.ndarray:
        """Normalize attribution map to [0, 1] range."""
        attr_min = attr_map.min()
        attr_max = attr_map.max()
        
        if attr_max - attr_min > 1e-8:
            return (attr_map - attr_min) / (attr_max - attr_min)
        else:
            return np.zeros_like(attr_map)


def visualize_attribution(
    input_image: np.ndarray,
    attribution_map: np.ndarray,
    title: str = "Attribution",
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize attribution map overlaid on the input image.
    
    Args:
        input_image: Original image as numpy array (H, W, C)
        attribution_map: Attribution map (H, W)
        title: Plot title
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(input_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attribution heatmap
    im = axes[1].imshow(attribution_map, cmap='hot')
    axes[1].set_title(f"{title} Attribution")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(input_image)
    overlay = axes[2].imshow(attribution_map, cmap='hot', alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    plt.colorbar(overlay, ax=axes[2], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_all_methods(
    input_image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    part_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Visualize attributions from all methods in a single figure.
    
    Args:
        input_image: Original image as numpy array (H, W, C)
        attributions: Dictionary of attribution maps
        part_mask: Optional ground truth part mask
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    n_methods = len(attributions) + (1 if part_mask is not None else 0) + 1
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(input_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth part mask if available
    idx = 1
    if part_mask is not None:
        axes[idx].imshow(input_image)
        axes[idx].imshow(part_mask, cmap='Greens', alpha=0.5)
        axes[idx].set_title("Ground Truth Parts")
        axes[idx].axis('off')
        idx += 1
    
    # Attribution maps
    for method_name, attr_map in attributions.items():
        if attr_map is not None:
            # Normalize for visualization
            attr_norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
            
            axes[idx].imshow(input_image)
            im = axes[idx].imshow(attr_norm, cmap='hot', alpha=0.6)
            axes[idx].set_title(method_name.replace('_', ' ').title())
            axes[idx].axis('off')
        else:
            axes[idx].set_title(f"{method_name} (failed)")
            axes[idx].axis('off')
        idx += 1
    
    # Hide remaining axes
    for i in range(idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Test explainability methods
    import torchvision.transforms as transforms
    from .model import create_model
    from .data_loader import CUB200Dataset, get_transforms
    
    print("Testing explainability methods...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a dummy model (not trained, just for testing)
    model = create_model('resnet50', num_classes=200, pretrained=True, device=device)
    
    # Load a test image
    dataset = CUB200Dataset(is_train=False, transform=get_transforms(False), load_parts=True)
    sample = dataset[0]
    
    input_tensor = sample['image'].unsqueeze(0)
    target_class = sample['label']
    
    print(f"Testing with class {target_class}")
    
    # Initialize explainability methods
    explainer = ExplainabilityMethods(model, device)
    
    # Test each method
    print("\nTesting Integrated Gradients...")
    ig_attr = explainer.get_integrated_gradients(input_tensor, target_class)
    print(f"  Shape: {ig_attr.shape}")
    
    print("\nTesting Input Gradients...")
    input_grad_attr = explainer.get_input_gradients(input_tensor, target_class)
    print(f"  Shape: {input_grad_attr.shape}")
    
    print("\nTesting Saliency...")
    sal_attr = explainer.get_saliency(input_tensor, target_class)
    print(f"  Shape: {sal_attr.shape}")
    
    print("\nTesting LIME (this may take a while)...")
    lime_attr = explainer.get_lime(input_tensor, target_class, n_samples=100)
    print(f"  Shape: {lime_attr.shape}")
    
    print("\nAll tests passed!")
