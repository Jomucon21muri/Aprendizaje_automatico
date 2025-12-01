# Tarea: Large Language Models (LLM)
## Sistemas de Aprendizaje Automático - Bloque 6

### Objetivo
Aprender a trabajar con LLMs: desde prompting hasta fine-tuning en aplicaciones prácticas.

### Actividades

#### Actividad 1: Exploración de APIs de LLMs
- Registra en: OpenAI, Anthropic, o Google Cloud
- Experimenta con diferentes modelos
- Compara capacidades y costos
- Crea ejemplos zero-shot y few-shot

#### Actividad 2: Prompt Engineering
- Diseña prompts efectivos para tareas:
  - Clasificación de sentimientos
  - Resumen de textos
  - Extracción de entidades
  - Generación de código
- Documenta técnicas: role-based, step-by-step, ejemplos
- Compara resultados con diferentes prompts

#### Actividad 3: Chain-of-Thought y Reasoning
- Implementa razonamiento paso-a-paso
- Self-consistency: Múltiples caminos
- Analiza mejora en complejos problemas
- Visualiza procesos de pensamiento

#### Actividad 4: Retrieval-Augmented Generation (RAG)
- Implementa pipeline RAG:
  1. Vector database (Pinecone, Chroma)
  2. Embed documentos
  3. Consulta + Recuperación
  4. LLM genera respuesta contextualizada
- Compara con LLM sin contexto
- Mide reducción de hallucinations

#### Actividad 5: Fine-tuning o LoRA
- Opción A: Fine-tuning completo de LLaMA/Mistral
- Opción B: LoRA en modelo base
- Dataset: Especialidad específica
- Comparar: Base vs Fine-tuned rendimiento
- Análisis: Cuándo usar cada enfoque

#### Actividad 6: Aplicación Integral
Desarrolla aplicación completa:
1. **Entrada**: Usuario proporciona consulta o documento
2. **Procesamiento**: Prompt construction, RAG si aplica
3. **LLM Call**: API o modelo local
4. **Post-procesamiento**: Validación, extracción
5. **Interfaz**: CLI, web, o API
6. **Evaluación**: Métricas y feedback humano

### Criterios de Evaluación
- Exploración de APIs: 10%
- Prompt Engineering: 20%
- Chain-of-Thought: 15%
- RAG Implementation: 20%
- Fine-tuning/LoRA: 15%
- Aplicación integral: 20%

### Entrega
- Notebook con experimentos (Colab)
- Scripts de prompting optimizados
- Implementación RAG funcional
- Modelo fine-tuned (si aplica)
- Aplicación con interfaz
- Documento: mejores prácticas y lecciones aprendidas
- Análisis: costos, latencia, calidad