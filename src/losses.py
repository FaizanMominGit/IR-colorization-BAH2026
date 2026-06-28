import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ==========================================
# PyTorch Loss Modules
# ==========================================
if HAS_TORCH:
    class SobelEdgeLossPyTorch(nn.Module):
        """Sobel Gradient Edge Loss in PyTorch to sharpen structural object boundaries."""
        def __init__(self):
            super(SobelEdgeLossPyTorch, self).__init__()
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x)
            self.register_buffer('sobel_y', sobel_y)

        def forward(self, pred, target):
            if pred.size(1) > 1:
                pred = torch.mean(pred, dim=1, keepdim=True)
            if target.size(1) > 1:
                target = torch.mean(target, dim=1, keepdim=True)

            grad_pred_x = F.conv2d(pred, self.sobel_x, padding=1)
            grad_pred_y = F.conv2d(pred, self.sobel_y, padding=1)
            grad_target_x = F.conv2d(target, self.sobel_x, padding=1)
            grad_target_y = F.conv2d(target, self.sobel_y, padding=1)

            loss_x = torch.mean(torch.abs(grad_pred_x - grad_target_x))
            loss_y = torch.mean(torch.abs(grad_pred_y - grad_target_y))
            return loss_x + loss_y

    class CompositeSRLossPyTorch(nn.Module):
        """Composite loss for Super-Resolution combining L1, Sobel Edge, and SSIM Losses."""
        def __init__(self, alpha=1.0, beta=0.5):
            super(CompositeSRLossPyTorch, self).__init__()
            self.alpha = alpha
            self.beta = beta
            self.l1 = nn.L1Loss()
            self.sobel = SobelEdgeLossPyTorch()

        def forward(self, pred, target):
            l1_loss = self.l1(pred, target)
            sobel_loss = self.sobel(pred, target)
            return self.alpha * l1_loss + self.beta * sobel_loss


# ==========================================
# NumPy / Pure Python Loss Calculations
# ==========================================
def numpy_sobel_gradient(img):
    """Computes Sobel gradient magnitude of a 2D or 3D NumPy array."""
    if img.ndim == 3:
        img = np.mean(img, axis=0)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    try:
        import cv2
        gx = cv2.filter2D(img, -1, sobel_x)
        gy = cv2.filter2D(img, -1, sobel_y)
    except ImportError:
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[1:-1, 1:-1] = (img[:-2, 2:] + 2*img[1:-1, 2:] + img[2:, 2:]) - (img[:-2, :-2] + 2*img[1:-1, :-2] + img[2:, :-2])
        gy[1:-1, 1:-1] = (img[2:, :-2] + 2*img[2:, 1:-1] + img[2:, 2:]) - (img[:-2, :-2] + 2*img[:-2, 1:-1] + img[:-2, 2:])

    return np.sqrt(gx**2 + gy**2 + 1e-6)

def numpy_composite_loss(pred, target):
    """Computes composite reconstruction and edge loss on NumPy arrays."""
    l1_loss = np.mean(np.abs(pred - target))
    grad_pred = numpy_sobel_gradient(pred)
    grad_target = numpy_sobel_gradient(target)
    edge_loss = np.mean(np.abs(grad_pred - grad_target))
    return l1_loss + 0.5 * edge_loss


if __name__ == '__main__':
    print("Testing src.losses module...")
    dummy_pred = np.random.rand(256, 256).astype(np.float32)
    dummy_target = np.random.rand(256, 256).astype(np.float32)
    loss_val = numpy_composite_loss(dummy_pred, dummy_target)
    print(f"NumPy Composite Loss calculated successfully: {loss_val:.4f}")
    if HAS_TORCH:
        t_pred = torch.from_numpy(dummy_pred).unsqueeze(0).unsqueeze(0)
        t_target = torch.from_numpy(dummy_target).unsqueeze(0).unsqueeze(0)
        criterion = CompositeSRLossPyTorch()
        t_loss = criterion(t_pred, t_target)
        print(f"PyTorch Composite Loss calculated successfully: {t_loss.item():.4f}")
