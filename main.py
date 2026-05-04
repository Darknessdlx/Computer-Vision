# ==========================
# Proyecto: Visión por Computadora con DeepFace
# ==========================

# Instalación de dependencias (solo ejecutar una vez)
# !pip install deepface opencv-python matplotlib

from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Cargar imagen o usar cámara
# Para pruebas, puedes usar imágenes locales en la carpeta 'images/'
# Ejemplo: img_path = "./images/persona1.jpg"
img_path = "./images/persona1.jpg"

# Mostrar imagen
img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Análisis facial
analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'], enforce_detection=False)
print("Análisis facial:")
print(analysis)

# Reconocimiento facial (comparar con otra imagen)
# Ejemplo: comparar persona1 con persona2
result = DeepFace.verify(img1_path="./images/persona1.jpg",
                         img2_path="./images/persona2.jpg",
                         enforce_detection=False)
print("¿Es la misma persona?:", result['verified'])

# Simulación de funcionalidad (Proyecto 3: Hogar inteligente)
if result['verified']:
    print("Bienvenido, Bryan. Ajustando luces y música relajante...")
else:
    print("Usuario no reconocido. Activando modo invitado.")
