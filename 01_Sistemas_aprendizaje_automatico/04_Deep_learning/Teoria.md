# Contenido Teórico: Deep Learning
## Sistemas de Aprendizaje Automático - Bloque 4

## 1. Fundamentos de Deep Learning

### 1.1 Diferencia con Shallow Learning
- Shallow: 1-2 capas ocultas
- Deep: 3+ capas, mayor representatividad
- Feature learning automático vs manual

### 1.2 Representación Jerárquica
- Capas inferiores: características simples (bordes, texturas)
- Capas intermedias: características complejas
- Capas superiores: conceptos abstractos

## 2. Redes Convolucionales (CNN)

### 2.1 Operación Convolucional
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

### 2.3 Arquitecturas Clásicas
- **LeNet-5**: Primeras CNNs (dígitos)
- **AlexNet**: ImageNet 2012 (breakthrough)
- **VGGNet**: Arquitectura profunda simple
- **ResNet**: Conexiones residuales (152 capas)
- **Inception**: Multi-escala de características

### 2.4 Aplicaciones
- Clasificación de imágenes
- Detección de objetos (YOLO, R-CNN)
- Segmentación semántica
- Reconocimiento facial

## 3. Redes Recurrentes (RNN)

### 3.1 RNN Básico
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

## 5. Redes Generativas Adversariales (GANs)

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

## 6. Técnicas de Optimización

### 6.1 Transfer Learning
- Usar modelos preentrenados (ImageNet)
- Fine-tuning en dataset específico
- Ahorro de datos y tiempo

### 6.2 Data Augmentation
- Rotación, traslación, zoom
- Mixup, Cutmix
- Aumenta tamaño efectivo dataset

### 6.3 Regularización Avanzada
- Batch Normalization: Normaliza activaciones
- Dropout: Desactiva neuronas aleatoriamente
- Layer Normalization, Group Normalization
- Weight decay (L2 regularization)