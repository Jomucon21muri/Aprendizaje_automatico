# Tarea: transformers
## Sistemas de aprendizaje automático - Bloque 5

### Objetivo
Implementar y aplicar modelos Transformer para tareas de NLP avanzadas.

### Actividades

#### Actividad 1: análisis de Attention Mechanism
- Implementa self-attention escalado manualmente
- Visualiza attention weights
- Explica qué palabras atienden qué
- Compara con multi-head attention

#### Actividad 2: fine-tuning de BERT
- Selecciona tarea: clasificación, NER, QA
- Carga BERT preentrenado (Hugging Face)
- Adapta cabeza para tarea específica
- Entrena con datos específicos del dominio
- Evalúa rendimiento

#### Actividad 3: comparación modelos
- BERT vs DistilBERT: tamaño, velocidad, precisión
- BERT vs GPT-2 en generación
- Análisis coste-beneficio
- Recomendación por caso de uso

#### Actividad 4: generación con GPT-2/GPT-3 API
- Implementa prompts efectivos
- Few-shot learning
- Chain-of-Thought prompting
- Analiza calidad de generación

#### Actividad 5: Vision Transformer (ViT)
- Implementa ViT simple o usa preentrenado
- Clasificación de imágenes
- Compara con CNN tradicional
- Análisis de attention en imágenes

#### Actividad 6: aplicación completa
Crea aplicación pipeline:
1. Entrada de usuario (texto/imagen)
2. Preprocesamiento
3. Modelo Transformer
4. Post-procesamiento
5. Presentación de resultados

### Criterios de evaluación
- Entendimiento de Attention: 15%
- Fine-tuning BERT: 20%
- Comparación modelos: 15%
- Generación de texto: 15%
- Vision Transformer: 15%
- Aplicación completa: 20%

### Entrega
- Notebook Jupyter con implementaciones
- Visualizaciones de attention weights
- Modelos fine-tuned
- Comparativa de resultados
- Aplicación funcional (script .py)
- Reporte de análisis y conclusiones