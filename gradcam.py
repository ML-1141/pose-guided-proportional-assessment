from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import SiameseSTN_AMCNN


try:
    import cv2
except Exception:
    cv2 = None

try:
    from pytorch_grad_cam import GradCAM, EigenCAM
    _HAS_CAM = True
except Exception:
    _HAS_CAM = False

class CosineSimilarityTarget:
    """
    For Grad-CAM: scalar target = cosine similarity to reference embedding.
    """
    def __init__(self, reference_embedding: torch.Tensor):
        if reference_embedding.dim() == 2 and reference_embedding.size(0) == 1:
            reference_embedding = reference_embedding[0]
        self.ref = F.normalize(reference_embedding, p=2, dim=0).detach()

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        z = F.normalize(model_output, p=2, dim=1)
        sim = torch.matmul(z, self.ref.to(z.device))  # (B,)
        sim = torch.tanh(sim)  # keep gradients alive
        return sim


class GradCAMWrapper:
    """
    Grad-CAM for the learner branch, using cosine sim to standard as target.
    """
    def __init__(self, siamese_model: SiameseSTN_AMCNN, use_eigencam: bool = False):
        if not _HAS_CAM:
            raise ImportError(
                "pytorch-grad-cam is not installed. Install with:\n"
                "pip install pytorch-grad-cam"
            )
        self.model   = siamese_model
        self.encoder = siamese_model.encoder
        self.target_layers = [self.encoder.target_layer]

        self.cam = (EigenCAM if use_eigencam else GradCAM)(
            model=self.encoder,
            target_layers=self.target_layers,
        )

    @torch.no_grad()
    def reference_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        return self.model.encode_standard(image_tensor)

    def heatmap_vs_reference(
        self,
        image_tensor: torch.Tensor,      # learner, (1,C,H,W)
        reference_tensor: torch.Tensor,  # standard, (1,C,H,W)
    ) -> np.ndarray:
        ref_z = self.reference_embedding(reference_tensor)  # (1,D)
        targets = [CosineSimilarityTarget(ref_z)]

        if self.model.use_stn and self.model.stn is not None:
            learner_aligned = self.model.stn(image_tensor)
        else:
            learner_aligned = image_tensor

        grayscale = self.cam(input_tensor=learner_aligned, targets=targets)
        return grayscale[0]

    @staticmethod
    def overlay_on_image(heatmap: np.ndarray, image_bgr: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        if cv2 is None:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
        h, w = image_bgr.shape[:2]
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
        heatmap_color = cv2.applyColorMap(cv2.resize(heatmap_uint8, (w, h)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
        return overlay


