# Contenido teórico: deep learning
## Sistemas de aprendizaje automático - Bloque 4

## 1. Fundamentos de deep learning

### 1.1 Diferencia con shallow learning
- Shallow: 1-2 capas ocultas
- Deep: 3+ capas, mayor representatividad
- Feature learning automático vs manual

### 1.2 Representación jerárquica
- Capas inferiores: características simples (bordes, texturas)
- Capas intermedias: características complejas
- Capas superiores: conceptos abstractos

## 2. Redes convolucionales (CNN)

### 2.1 Operación convolucional
- Filtros/kernels que se deslizan sobre entrada
- Extrae características locales
- Reducción de parámetros vs redes fully-connected

### 2.2 Arquitectura CNN
```
Entrada → Conv → ReLU → Pool → ... → FC → Output
```
- Convolución: Feature extraction
- Pooling: Reducción de dimensionalidad
- FC: Clasificación final

### 2.3 Arquitecturas clásicas
- **LeNet-5**: primeras CNNs (dígitos)
- **AlexNet**: ImageNet 2012 (breakthrough)
- **VGGNet**: arquitectura profunda simple
- **ResNet**: conexiones residuales (152 capas)
- **Inception**: multi-escala de características

### 2.4 Aplicaciones
- Clasificación de imágenes
- Detección de objetos (YOLO, R-CNN)
- Segmentación semántica
- Reconocimiento facial

## 3. Redes recurrentes (RNN)

### 3.1 RNN básico
- Conexiones recurrentes (retroalimentación)
- Memoria de estados pasados
- Vanishing/Exploding gradient problem

### 3.2 LSTM (Long Short-Term Memory)
- Gated mechanisms (forget, input, output gates)
- Resuelve problema de gradiente
- Memoria a largo plazo

### 3.3 GRU (Gated Recurrent Unit)
- Simplificación de LSTM
- Menos parámetros
- Similar rendimiento en muchas tareas

### 3.4 Aplicaciones
- Predicción de series temporales
- Traducción automática (Seq2Seq)
- Análisis de sentimiento
- Generación de texto

## 4. Autoencoders

### 4.1 Arquitectura
- **Encoder**: Comprime entrada a representación latente
- **Decoder**: Reconstruye desde representación
- Aprendizaje no supervisado

### 4.2 Variantes
- **Stacked Autoencoders**: Múltiples capas
- **Denoising AE**: Robustez a ruido
- **Variational AE (VAE)**: Distribución latente probabilística

### 4.3 Aplicaciones
- Reducción de dimensionalidad
- Compresión de datos
- Detección de anomalías
- Generación de datos nuevos

## 5. Redes generativas adversariales (GANs)

### 5.1 Arquitectura
- **Generator**: Crea datos sintéticos
- **Discriminator**: Clasifica real vs sintético
- Juego de minimax entre ambas redes

### 5.2 Variantes
- **DCGAN**: Deep Convolutional GAN
- **CycleGAN**: Conversión entre dominios
- **StyleGAN**: Control fino de generación
- **Diffusion Models**: Generación iterativa

### 5.3 Aplicaciones
- Síntesis de imágenes
- Super-resolución
- Transferencia de estilo
- Generación de rostros/escenas

## 6. Técnicas de optimización

### 6.1 Transfer Learning
- Usar modelos preentrenados (ImageNet)
- Fine-tuning en dataset específico
- Ahorro de datos y tiempo

### 6.2 Data Augmentation
- Rotación, traslación, zoom
- Mixup, Cutmix
- Aumenta tamaño efectivo dataset

### 6.3 Regularización avanzada
- Batch Normalization: normaliza activaciones
- Dropout: Desactiva neuronas aleatoriamente
- Layer Normalization, Group Normalization
- Weight decay (L2 regularization)