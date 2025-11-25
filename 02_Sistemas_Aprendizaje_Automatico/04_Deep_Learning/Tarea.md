# Tarea: Deep Learning
## Sistemas de Aprendizaje Automático - Bloque 4

### Objetivo
Implementar y comparar arquitecturas de Deep Learning en problemas reales de visión computacional.

### Actividades

#### Actividad 1: CNN para Clasificación de Imágenes
- Selecciona dataset (CIFAR-10, MNIST, STL-10)
- Diseña arquitectura CNN personalizada (3-5 capas)
- Implementa con TensorFlow/Keras o PyTorch
- Entrena y evalúa rendimiento

#### Actividad 2: Transfer Learning
- Carga modelo preentrenado (ResNet, VGG, Inception)
- Adapta última capa(s) al problema
- Fine-tune con learning rate bajo
- Compara: modelo personalizado vs transfer learning

#### Actividad 3: Data Augmentation
- Implementa transformaciones (rotación, flip, etc.)
- Compara rendimiento: con vs sin augmentation
- Analiza impacto en generalización
- Crea visualizaciones de ejemplos aumentados

#### Actividad 4: Detección de Anomalías con Autoencoder
- Entrena Autoencoder en datos normales
- Usa error de reconstrucción como métrica
- Detecta ejemplos anómalos
- Evalúa precisión y recall

#### Actividad 5: Generación de Imágenes (GAN Simple)
- Implementa GAN simple o DCGAN
- Entrena en dataset (MNIST, CelebA)
- Genera imágenes nuevas
- Visualiza progresión del entrenamiento

#### Actividad 6: Análisis y Documentación
- Compara arquitecturas: velocidad, precisión, parámetros
- Crea tabla de resultados
- Analiza overfitting/underfitting
- Propone mejoras futuras

### Criterios de Evaluación
- Implementación CNN: 20%
- Transfer Learning: 15%
- Data Augmentation: 15%
- Autoencoder: 15%
- GAN: 15%
- Análisis y documentación: 20%

### Entrega
- Notebook Jupyter comentado
- Modelos entrenados (.h5, .pt)
- Gráficos: pérdida, ejemplos generados
- Tabla comparativa de arquitecturas
- Reporte técnico con recomendaciones