"""
ADDNet2D com backbone Xception para detecção de deepfakes.
Baseado no paper que usa blocos de injeção de atenção (ADD Blocks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ADDBlock(nn.Module):
    """
    Bloco de injeção de atenção.
    Multiplica as features pela máscara redimensionada.
    """
    def __init__(self):
        super(ADDBlock, self).__init__()

    def forward(self, feature_map, attention_mask):
        # Redimensiona a máscara para o tamanho atual do feature map
        target_size = feature_map.shape[2:] 
        scaled_mask = F.adaptive_avg_pool2d(attention_mask, target_size)
        
        # Multiplicação Element-wise (Ajuste de pesos via Atenção)
        return feature_map * scaled_mask


class ADDNet2D_Xception(nn.Module):
    """
    ADDNet2D com backbone Xception pré-treinada.
    Usa ADD Blocks para injetar atenção baseada em landmarks faciais.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ADDNet2D_Xception, self).__init__()
        
        # 1. Carrega a Xception pré-treinada do timm
        self.backbone = timm.create_model('legacy_xception', pretrained=pretrained)
        
        # 2. Desmonta a Xception para inserir os ADD Blocks
        # Stem (Entrada inicial)
        self.stem = nn.Sequential(
            self.backbone.conv1, self.backbone.bn1, self.backbone.act1,
            self.backbone.conv2, self.backbone.bn2, self.backbone.act2
        )
        
        # Entry Flow (blocks 1-3)
        self.block1 = self.backbone.block1
        self.block2 = self.backbone.block2
        self.block3 = self.backbone.block3
        
        # Middle Flow (blocks 4-11)
        self.middle_flow = nn.Sequential(
            self.backbone.block4,
            self.backbone.block5,
            self.backbone.block6,
            self.backbone.block7,
            self.backbone.block8,
            self.backbone.block9,
            self.backbone.block10,
            self.backbone.block11,
        )
        
        # Exit Flow (block 12)
        self.exit_flow = self.backbone.block12
        
        # Final layers
        self.conv3 = self.backbone.conv3
        self.bn3 = self.backbone.bn3
        self.act3 = self.backbone.act3
        self.conv4 = self.backbone.conv4
        self.bn4 = self.backbone.bn4
        self.act4 = self.backbone.act4
        
        # 3. Módulo de injeção de atenção
        self.add_handler = ADDBlock()
        
        # 4. Classificador Final
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, image, attention_mask):
        """
        Forward pass com injeção de atenção.
        
        Args:
            image: Tensor de imagens [B, 3, H, W]
            attention_mask: Tensor de máscaras [B, 1, H, W]
        
        Returns:
            Logits de classificação [B, num_classes]
        """
        # --- Stage 0: Stem ---
        x = self.stem(image)
        
        # --- Stage 1: Entry Flow (com injeção) ---
        x = self.block1(x)
        x = self.add_handler(x, attention_mask)
        
        x = self.block2(x)
        x = self.add_handler(x, attention_mask)
        
        x = self.block3(x)
        x = self.add_handler(x, attention_mask)
        
        # --- Stage 2: Middle Flow (com injeção) ---
        x = self.middle_flow(x)
        x = self.add_handler(x, attention_mask)
        
        # --- Stage 3: Exit Flow ---
        x = self.exit_flow(x)
        x = self.add_handler(x, attention_mask)
        
        # --- Stage 4: Final Feature Extraction ---
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
    # Batch de 4 imagens, 3 canais, 299x299 (padrão Xception)
    img = torch.randn(4, 3, 299, 299)
    mask = torch.rand(4, 1, 299, 299)
    
    model = ADDNet2D_Xception(num_classes=2, pretrained=False)
    
    output = model(img, mask)
    print("Saída do modelo:", output.shape)  # Esperado: [4, 2]
    probs = F.softmax(output, dim=1)
    print("Probabilidades:", probs)