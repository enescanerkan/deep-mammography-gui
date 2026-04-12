"""
Model manager for loading and managing MLO and CC landmark detection models.
"""

import os
import torch
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from networks.landmark_models import CRAUNet, CCCRAUNet


class IModelManager(ABC):
    """Interface for model managers."""
    
    @abstractmethod
    def load_models(self) -> None:
        """Load all required models."""
        pass
    
    @abstractmethod
    def predict_landmarks(self, image: np.ndarray, model_type: str) -> Optional[np.ndarray]:
        """Predict landmarks for the given image."""
        pass


class ModelManager(IModelManager):
    """Manager for loading and using MLO and CC landmark detection models."""
    
    def __init__(self):
        """Initialize the model manager."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}
        
    def load_models(self) -> None:
        """Load MLO and CC models."""
        self._load_mlo_model()
        self._load_cc_model()
    
    def _load_mlo_model(self) -> None:
        """Load MLO model."""
        model_path = self._get_model_path("mlo_craunet.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MLO model not found: {model_path}")
        
        # MLO: 3 landmarks (2 pectoral + 1 nipple) = 6 outputs
        model = CRAUNet(in_channels=1, out_features=6).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        self.models['mlo'] = model
        
    def _load_cc_model(self) -> None:
        """Load CC model."""
        model_path = self._get_model_path("cc_craunet.pth")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CC model not found: {model_path}")
        
        # CC: 1 landmark (nipple) = 2 outputs
        model = CCCRAUNet(in_channels=1, out_features=2).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        
        self.models['cc'] = model
    
    def _get_model_path(self, model_filename: str) -> str:
        """Get full path to model file.
        
        Supports both development mode and PyInstaller bundled executable.
        """
        import sys
        
        # Check if running as PyInstaller bundle
        if getattr(sys, 'frozen', False):
            # Running as bundled executable
            base_path = sys._MEIPASS
        else:
            # Running in development mode
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.dirname(os.path.dirname(current_dir))
        
        return os.path.join(base_path, "weights", model_filename)

    def predict_landmarks(self, image: np.ndarray, model_type: str, 
                         original_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Predict landmarks for the given image.
        
        Args:
            image: Input image array (should be 512x512 preprocessed)
            model_type: Type of model ('mlo' or 'cc')
            original_shape: Not used, kept for compatibility
            
        Returns:
            Predicted landmarks as numpy array in 512x512 space
        """
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not available")
        
        try:
            model = self.models[model_type]
            
            if len(image.shape) == 3:
                image = image[0]
            
            if image.shape != (512, 512):
                print(f"Warning: Image shape is {image.shape}, resizing to (512, 512)...")
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            
            img_tensor = torch.from_numpy(image).float()
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0
            
            with torch.no_grad():
                predictions = model(img_tensor).cpu().numpy().flatten()
            
            if model_type == 'mlo':
                # MLO: 6 values -> reshape to (3, 2) for 3 landmarks
                predicted_landmarks = np.reshape(predictions, (-1, 2))
                predicted_landmarks[:, 0] *= 512
                predicted_landmarks[:, 1] *= 512
            else:
                # CC: 2 values -> (x, y) for nipple
                pred_x = predictions[0] * 512
                pred_y = predictions[1] * 512
                predicted_landmarks = np.array([[pred_x, pred_y]])
            
            return predicted_landmarks
                
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded."""
        return model_type in self.models
    
    def get_available_models(self) -> list:
        """Get list of available model types."""
        return list(self.models.keys())
