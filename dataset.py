"""
Dataset para treinamento do ADDNet com geração de máscaras de atenção.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import dlib


class DeepfakeDataset(Dataset):
    """
    Dataset para detecção de deepfakes com geração de máscaras de atenção.
    
    Estrutura esperada:
        root/
            train/
                fake/
                real/
            test/
                fake/
                real/
    """
    def __init__(self, root_dir, split='train', transform=None, image_size=299):
        """
        Args:
            root_dir: Caminho para o diretório do dataset
            split: 'train' ou 'test'
            transform: Transformações a aplicar nas imagens
            image_size: Tamanho da imagem de saída (padrão 299 para Xception)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        # Inicializa detector de landmarks
        self.detector = dlib.get_frontal_face_detector()
        predictor_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "shape_predictor_68_face_landmarks.dat"
        )
        if os.path.exists(predictor_path):
            self.predictor = dlib.shape_predictor(predictor_path)
        else:
            print(f"Aviso: shape_predictor_68_face_landmarks.dat não encontrado em {predictor_path}")
            self.predictor = None
        
        # Carrega lista de imagens
        self.samples = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split)
        
        # Classe 0 = real, Classe 1 = fake
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Aviso: Diretório não encontrado: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)
        
        print(f"[{split.upper()}] Total de amostras: {len(self.samples)}")
        print(f"  - Real: {self.labels.count(0)}")
        print(f"  - Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def shape_to_np(self, shape):
        """Converte landmarks dlib para numpy array."""
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def generate_attention_mask(self, image_cv):
        """
        Gera máscara de atenção baseada em landmarks faciais.
        
        Returns:
            Máscara normalizada [0, 1] com pesos maiores para face e órgãos.
        """
        h, w = image_cv.shape[:2]
        
        # Se não tiver predictor, retorna máscara uniforme
        if self.predictor is None:
            return np.ones((h, w), dtype=np.float32)
        
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        
        # Se não encontrar face, retorna máscara uniforme
        if len(rects) == 0:
            return np.ones((h, w), dtype=np.float32)
        
        shape = self.predictor(gray, rects[0])
        landmarks = self.shape_to_np(shape)
        
        # Cria máscara da face (contorno)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(landmarks)
        cv2.fillConvexPoly(face_mask, hull, 255)
        
        # Cria máscara dos órgãos (olhos, nariz, boca)
        organ_mask = np.zeros((h, w), dtype=np.uint8)
        organs = [
            range(36, 42),  # Olho Direito
            range(42, 48),  # Olho Esquerdo
            range(27, 36),  # Nariz
            range(48, 68)   # Boca
        ]
        for idx in organs:
            organ_hull = cv2.convexHull(landmarks[list(idx)])
            cv2.fillConvexPoly(organ_mask, organ_hull, 255)
        
        # Aplica blur gaussiano para suavizar
        k_size = (15, 15)
        face_mask_blurred = cv2.GaussianBlur(face_mask, k_size, 0)
        organ_mask_blurred = cv2.GaussianBlur(organ_mask, k_size, 0)
        
        # Combina máscaras
        f_mask = face_mask_blurred.astype(np.float32) / 255.0
        o_mask = organ_mask_blurred.astype(np.float32) / 255.0
        
        final_attention = f_mask + o_mask
        
        # Normaliza para [0, 1]
        final_attention = cv2.normalize(
            final_attention, None, 
            alpha=0, beta=1, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_32F
        )
        
        return final_attention
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Carrega imagem com OpenCV para gerar máscara
        image_cv = cv2.imread(img_path)
        if image_cv is None:
            print(f"Erro ao carregar: {img_path}")
            # Retorna tensor vazio em caso de erro
            return torch.zeros(3, self.image_size, self.image_size), \
                   torch.ones(1, self.image_size, self.image_size), \
                   torch.tensor(label, dtype=torch.long)
        
        # Gera máscara de atenção
        attention_mask = self.generate_attention_mask(image_cv)
        
        # Redimensiona imagem e máscara
        image_cv = cv2.resize(image_cv, (self.image_size, self.image_size))
        attention_mask = cv2.resize(attention_mask, (self.image_size, self.image_size))
        
        # Converte para PIL para aplicar transforms
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        # Aplica transformações na imagem
        if self.transform:
            image = self.transform(image_pil)
        else:
            # Conversão padrão para tensor
            image = torch.from_numpy(
                np.array(image_pil).transpose(2, 0, 1)
            ).float() / 255.0
        
        # Converte máscara para tensor [1, H, W]
        mask_tensor = torch.from_numpy(attention_mask).unsqueeze(0).float()
        
        return image, mask_tensor, torch.tensor(label, dtype=torch.long)


def get_dataloaders(root_dir, batch_size=16, num_workers=4, image_size=299):
    """
    Cria dataloaders de treino e teste.
    
    Args:
        root_dir: Caminho para o dataset
        batch_size: Tamanho do batch
        num_workers: Número de workers para carregar dados
        image_size: Tamanho da imagem
    
    Returns:
        train_loader, test_loader
    """
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Transformações para treino (com augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Transformações para teste (sem augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = DeepfakeDataset(
        root_dir, split='train', 
        transform=train_transform, 
        image_size=image_size
    )
    
    test_dataset = DeepfakeDataset(
        root_dir, split='test', 
        transform=test_transform, 
        image_size=image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Teste do dataset
    dataset = DeepfakeDataset(
        "/home/tasso/FR/our_model/deepfake_dataset/processed",
        split='train'
    )
    
    if len(dataset) > 0:
        image, mask, label = dataset[0]
        print(f"Imagem shape: {image.shape}")
        print(f"Máscara shape: {mask.shape}")
        print(f"Label: {label}")
