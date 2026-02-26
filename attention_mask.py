# attention mask module
import dlib
import cv2
import numpy as np

# entrada das imagens no formato do opencv (height, width, channels) and channels (BGR)
class AttentionMask():
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    def predict_landmarks(self, image):
        detected = self.detector(image)
        if len(detected) == 0:
            return None
        return self.predictor(image, detected[0])
    
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def align(self, image, landmarks):
        coords = self.shape_to_np(landmarks)

        # olhos na visao da pessoa
        (lstart, lend) = (42, 48)
        (rstart, rend) = (36, 42)

        leftEye = coords[lstart:lend]
        rightEye = coords[rstart:rend]

        leftEyeCenter = leftEye.mean(axis=0).astype("int")
        rightEyeCenter = rightEye.mean(axis=0).astype("int")

        dy = rightEyeCenter[1] - leftEyeCenter[1]
        dx = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dy, dx))

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated

    def generate_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)
        if len(rects) == 0:
            print("Nenhum rosto detectado.")
            return None, None, None
        
        rects = rects[0]
        
        landmarks_raw = self.predictor(gray, rects)

        aligned = self.align(image, landmarks_raw)      

        gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        rects_aligned = self.detector(gray_aligned, 1)
        if len(rects_aligned) == 0:
            print("Nenhum rosto detectado.")
            return None, None, None
        else:
            shape = predictor(gray_aligned, rects_aligned[0])
            

        rects_aligned = rects_aligned[0]
        landmarks_aligned = self.predictor(gray_aligned, rects_aligned)
        landmarks_aligned = self.shape_to_np(landmarks_aligned)
        
        landmarks = self.shape_to_np(shape)
        h, w = aligned.shape[:2]

        face_mask = np.zeros((h, w), dtype="uint8")
        hull = cv2.convexHull(landmarks)

        cv2.fillConvexPoly(face_mask, hull, 255)

        organ_mask = np.zeros((h, w), dtype="uint8")

        organs = [
            range(36, 42), # Olho Direito
            range(42, 48), # Olho Esquerdo
            range(27, 36), # Nariz
            range(48, 68)  # Boca
        ]
        
        for idx in organs:
            hull = cv2.convexHull(landmarks[idx])
            cv2.fillConvexPoly(organ_mask, hull, 255)

        k_size = (15, 15)
        face_mask_blurred = cv2.GaussianBlur(face_mask, k_size, 0)
        organ_mask_blurred = cv2.GaussianBlur(organ_mask, k_size, 0)

        f_mask = face_mask_blurred.astype("float32") / 255.0
        o_mask = organ_mask_blurred.astype("float32") / 255.0

        final_attention = f_mask + o_mask

        final_attention = cv2.normalize(final_attention, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return aligned_img, face_mask, organ_mask, final_attention
