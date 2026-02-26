"""
ADDNet2D com backbone Xception + BandpassLayer treinável.
A BandpassLayer filtra a imagem no domínio espacial antes de alimentar o ADDNet2D.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# =============================================================================
# BandpassLayer (copiado de our_model/our_model.py)
# =============================================================================

def get_gaussian_kernel_2d(ksize: int, sigma: torch.Tensor, device: torch.device):
    if ksize % 2 == 0:
        ksize += 1
    coords = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


class BandpassLayer(nn.Module):
    def __init__(self, kernel_size=15, initial_sigma_low=1.0, initial_gap=0.5):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma_base = nn.Parameter(torch.tensor(initial_sigma_low, dtype=torch.float32))
        self.gap = nn.Parameter(torch.tensor(initial_gap, dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        s_small = torch.abs(self.sigma_base) + 1e-5
        s_large = s_small + torch.abs(self.gap) + 1e-5

        k_small = get_gaussian_kernel_2d(self.kernel_size, s_small, device)
        k_large = get_gaussian_kernel_2d(self.kernel_size, s_large, device)

        k_small = k_small.view(1, 1, self.kernel_size, self.kernel_size).repeat(C, 1, 1, 1)
        k_large = k_large.view(1, 1, self.kernel_size, self.kernel_size).repeat(C, 1, 1, 1)

        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')

        blur_small = F.conv2d(x_padded, k_small, groups=C)
        blur_large = F.conv2d(x_padded, k_large, groups=C)
        result = blur_small - blur_large

        return result


# =============================================================================
# ADDBlock
# =============================================================================

class ADDBlock(nn.Module):
    """
    Bloco de injeção de atenção.
    Multiplica as features pela máscara redimensionada.
    """
    def __init__(self):
        super(ADDBlock, self).__init__()

    def forward(self, feature_map, attention_mask):
        target_size = feature_map.shape[2:]
        scaled_mask = F.adaptive_avg_pool2d(attention_mask, target_size)
        return feature_map * scaled_mask


# =============================================================================
# ADDNet2D_Xception_Bandpass
# =============================================================================

class ADDNet2D_Xception_Bandpass(nn.Module):
    """
    ADDNet2D com backbone Xception pré-treinada + BandpassLayer treinável.
    
    Pipeline:
        imagem -> BandpassLayer -> Xception + ADD Blocks -> classificação
    
    A BandpassLayer aprende qual banda de frequência espacial é mais
    discriminativa para a tarefa de deepfake detection.
    """
    def __init__(self, num_classes=2, pretrained=True,
                 kernel_size=31, initial_sigma_low=1.0, initial_gap=2.0):
        super(ADDNet2D_Xception_Bandpass, self).__init__()
        
        # 0. BandpassLayer treinável
        self.bandpass = BandpassLayer(
            kernel_size=kernel_size,
            initial_sigma_low=initial_sigma_low,
            initial_gap=initial_gap,
        )

        # 1. Carrega a Xception pré-treinada do timm
        backbone = timm.create_model('legacy_xception', pretrained=pretrained)

        # 2. Desmonta a Xception para inserir os ADD Blocks
        # Stem (Entrada inicial)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.act1,
            backbone.conv2, backbone.bn2, backbone.act2
        )

        # Entry Flow (blocks 1-3)
        self.block1 = backbone.block1
        self.block2 = backbone.block2
        self.block3 = backbone.block3

        # Middle Flow (blocks 4-11)
        self.middle_flow = nn.Sequential(
            backbone.block4,
            backbone.block5,
            backbone.block6,
            backbone.block7,
            backbone.block8,
            backbone.block9,
            backbone.block10,
            backbone.block11,
        )

        # Exit Flow (block 12)
        self.exit_flow = backbone.block12

        # Final layers
        self.conv3 = backbone.conv3
        self.bn3 = backbone.bn3
        self.act3 = backbone.act3
        self.conv4 = backbone.conv4
        self.bn4 = backbone.bn4
        self.act4 = backbone.act4

        # 3. Módulo de injeção de atenção
        self.add_handler = ADDBlock()

        # 4. Classificador Final
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, image, attention_mask):
        """
        Forward pass.

        Args:
            image: Tensor de imagens [B, 3, H, W]
            attention_mask: Tensor de máscaras [B, 1, H, W]

        Returns:
            Logits de classificação [B, num_classes]
        """
        # --- Stage 0: BandpassLayer ---
        x = self.bandpass(image)

        # --- Stage 1: Stem ---
        x = self.stem(x)

        # --- Stage 2: Entry Flow (com injeção) ---
        x = self.block1(x)
        x = self.add_handler(x, attention_mask)

        x = self.block2(x)
        x = self.add_handler(x, attention_mask)

        x = self.block3(x)
        x = self.add_handler(x, attention_mask)

        # --- Stage 3: Middle Flow (com injeção) ---
        x = self.middle_flow(x)
        x = self.add_handler(x, attention_mask)

        # --- Stage 4: Exit Flow ---
        x = self.exit_flow(x)
        x = self.add_handler(x, attention_mask)

        # --- Stage 5: Final Feature Extraction ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)

        # --- Classificação ---
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# --- TESTE RÁPIDO ---
if __name__ == "__main__":
    img = torch.randn(4, 3, 299, 299)
    mask = torch.rand(4, 1, 299, 299)

    model = ADDNet2D_Xception_Bandpass(num_classes=2, pretrained=False)

    output = model(img, mask)
    print("Saída do modelo:", output.shape)  # Esperado: [4, 2]
    probs = F.softmax(output, dim=1)
    print("Probabilidades:", probs)

    # Verifica parâmetros da bandpass
    bp = model.bandpass
    print(f"sigma_base: {bp.sigma_base.item():.4f}")
    print(f"gap: {bp.gap.item():.4f}")
    print(f"sigma_high: {bp.sigma_base.item() + bp.gap.item():.4f}")
