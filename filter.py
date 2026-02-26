import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from abc import ABC, abstractmethod

def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada em: {path}")
    return image

class Filter(ABC):
    @abstractmethod
    def __init__(self, raio_int: int, raio_ext: int, rows: int, cols: int):
        pass

class BandPassFilter(Filter):
    def __init__(self, raio_int: int, raio_ext: int, rows: int, cols: int):
        self.raio_int = raio_int if raio_int is not None else 0
        self.raio_ext = raio_ext if raio_ext is not None else rows
        self.rows = rows
        self.cols = cols
        center = (cols // 2, rows // 2)
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        D0 = (raio_ext + raio_int) / 2  
        W = raio_ext - raio_int         
        
        numerador = distance * W
        denominador = distance**2 - D0**2
        mask = np.ones_like(distance, dtype=np.float32)
        
        valid_mask = np.abs(denominador) >= 1e-10
        
        ratio = np.zeros_like(distance)
        ratio[valid_mask] = numerador[valid_mask] / denominador[valid_mask]
        
        if raio_int == 0:
            mask = np.exp(-(distance**2)/ (2*(raio_ext**2)))
        else:
            mask = np.exp(-(distance**2)/ (2*(raio_ext**2))) * (1 - np.exp(-(distance**2)/ (2*(raio_int**2))))
        
        self.mask = mask
        if raio_int != 0:
            self.mask[self.rows//2, self.cols//2] = 0

class FreqManager:
    def __init__(self, path: str = None, filter=None):
        self.path = path
        self.filter = filter

    def set_path(self, path: str):
        self.path = path

    def set_filter(self, raio: int, raio_ext: int = None, rows: int = 0, cols: int = 0):
        # Aqui instanciamos a sua classe de filtro Gaussiano
        self.filter = BandPassFilter(raio_int=raio, raio_ext=raio_ext, rows=rows, cols=cols)

    def transform_img(self):
        # Assumindo que load_image usa cv2.imread internamente (retorna BGR uint8)
        image = load_image(self.path) 
        image = cv2.resize(image, (256, 256))
        
        # O truque do padding reflexivo está perfeito para evitar artefatos de FFT
        image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT)
        
        # REMOVIDO: image = np.float32(image) 
        # Deixamos em uint8 para o cv2.cvtColor funcionar corretamente depois
        
        return image, image.shape[:2]

    def apply_filter(self, type: str, raio: int, raio_ext: int = None):
        image, (rows, cols) = self.transform_img()

        self.set_filter(raio=raio, raio_ext=raio_ext, rows=rows, cols=cols)
        
        # Prepara a máscara para 2 canais (Real e Imaginário do OpenCV)
        mask = self.filter.mask.astype(np.float32)
        mask_2_canais = cv2.merge([mask, mask])

        # Aplica conversão em uint8 (cores se mantêm corretas)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Transformada de Fourier: AQUI SIM convertemos o canal Y para float32
        v_float = np.float32(v)
        dft_v = cv2.dft(v_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        v_filt = np.fft.fftshift(dft_v)
        
        # Multiplicação elemento a elemento
        Ff = v_filt * mask_2_canais
        
        # Inversa
        f_ishift = np.fft.ifftshift(Ff)
        img_filt = cv2.idft(f_ishift)
        img_filt = cv2.magnitude(img_filt[:, :, 0], img_filt[:, :, 1])
        img_filt = cv2.normalize(img_filt, None, 0, 255, cv2.NORM_MINMAX)             
                   
        # Volta para uint8 (0 a 255)
        v_final = np.clip(img_filt, 0, 255).astype(np.uint8)
        
        # Recombina e volta para BGR
        hsv_filtered = cv2.merge((h, s, v_final))
        img_final_bgr = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)

        image_original = image[50:-50, 50:-50]
        img_final_bgr = img_final_bgr[50:-50, 50:-50]
        path_fold = f'data/{self.path.split("/")[-3]}/{self.path.split("/")[-2]}/{type}/{(str(self.filter.raio_int )+ "_") if self.filter.raio_int != 0 or type == "faixa" else ""}{self.filter.raio_ext if self.filter.raio_ext != 0 or type == "faixa" else ""}'
        if not os.path.exists(path_fold):
            os.makedirs(path_fold)

        cv2.imwrite(f'{path_fold}/{self.path.split("/")[-1].split(".")[0]}_filtered.png', img_final_bgr)

        # Retorna a imagem BGR padrão OpenCV para caso queira usar em outro lugar do código
        return img_final_bgr
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    fm = FreqManager()

    raios = [(0,10), (10,30), (30,50), (50,100)]
    types = ["faixa", "passa_alta", "passa_baixa"]

    for raio_par in raios:
        for typ in types:
            if typ == "faixa":
                raio = raio_par[0]
                raio_ext = raio_par[1]
            elif typ == "passa_alta":
                raio = raio_par[0]
                raio_ext = None
            elif typ == "passa_baixa":
                raio = None
                raio_ext = raio_par[1]

            for img in os.listdir(args.path):
                if img.endswith(".png"):
                    fm.set_path(args.path + "/" + img)
                    returned = fm.apply_filter(type=typ, raio=raio, raio_ext=raio_ext)
                print(returned.shape)