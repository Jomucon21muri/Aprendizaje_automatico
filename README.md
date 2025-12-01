# 📚 Curso de especialización en aprendizaje automático

Repositorio de formación integral en **machine learning y big data** con teoría, mapas conceptuales y ejercicios prácticos.

## 🎯 Competencia general

Programar y aplicar **sistemas inteligentes** que optimizan la gestión de la información y la explotación de datos masivos, garantizando:
- ✅ Acceso seguro a datos
- ✅ Criterios de accesibilidad y usabilidad
- ✅ Calidad según estándares establecidos
- ✅ Principios éticos y legales

---

## 📖 Módulos del curso

### 1. 🧠 [Sistemas de aprendizaje automático](./01_Sistemas_aprendizaje_automatico/)

**Objetivo**: Dominar técnicas de machine learning desde conceptos básicos hasta deep learning avanzado y aprendizaje por refuerzo.

#### Bloques de contenido:

| Bloque | Tema | Contenido |
|--------|------|----------|
| **0** | [Conceptos básicos de IA](./01_Sistemas_aprendizaje_automatico/00_Conceptos_basicos_ia/) | Definición, historia, tipos de IA, aplicaciones |
| **1** | [ML supervisado](./01_Sistemas_aprendizaje_automatico/01_Ml_supervisado/) | Clasificación, regresión, algoritmos clásicos |
| **2** | [ML no supervisado](./01_Sistemas_aprendizaje_automatico/02_Ml_no_supervisado/) | Clustering, reducción de dimensionalidad, anomalías |
| **3** | [Validación y evaluación](./01_Sistemas_aprendizaje_automatico/03_Validacion_evaluacion/) | Métricas, validación cruzada, ajuste de hiperparámetros |
| **4** | [Deep learning](./01_Sistemas_aprendizaje_automatico/04_Deep_learning/) | CNN, RNN, autoencoders, GANs, transfer learning |
| **5** | [Transformers](./01_Sistemas_aprendizaje_automatico/05_Transformers/) | Attention, BERT, GPT, ViT, arquitecturas modernas |
| **6** | [Large language models (LLM)](./01_Sistemas_aprendizaje_automatico/06_Llm/) | GPT, Claude, prompting, RAG, fine-tuning |
| **7** | [Reinforcement learning](./01_Sistemas_aprendizaje_automatico/07_Reinforcement_learning/) | Q-learning, policy gradient, actor-critic, DQN |
| **8** | [Sistemas expertos](./01_Sistemas_aprendizaje_automatico/08_Sistemas_expertos/) | Arquitectura, base de conocimiento, motor de inferencia |

Recursos por bloque:
- 📊 **Mapa_Conceptual.md**: Categorización de algoritmos
- 📖 **Teoria.md**: Fundamentos matemáticos y conceptuales
- 📝 **Tarea.md**: Implementación práctica

**Información adicional**: [Recursos complementarios](./01_Sistemas_aprendizaje_automatico/Informacion_Adicional.md)

---

### 2. 📊 [Big data](./02_Big_data/)

**Objetivo**: Gestionar y analizar datos masivos con tecnologías distribuidas modernas y técnicas de minería.

#### Bloques de contenido:

| Bloque | Tema | Contenido |
|--------|------|----------|
| **1** | [Fundamentos big data](./02_Big_data/01_Fundamentos_big_data/) | Las 5 Vs, arquitectura, tecnologías clave |
| **2** | [Herramientas procesamiento](./02_Big_data/02_Herramientas_procesamiento/) | Hadoop, Spark, stream processing, alternativas |
| **3** | [Análisis datos masivos](./02_Big_data/03_Analisis_datos_masivos/) | ETL, EDA, ML distribuido, visualización |
| **4** | [Minería de datos](./02_Big_data/04_Mineria_datos/) | CRISP-DM, técnicas predictivas, descriptivas, deployment |

Recursos por bloque:
- 📊 **Mapa_Conceptual.md**: Ecosistema de tecnologías
- 📖 **Teoria.md**: Conceptos y arquitecturas
- 📝 **Tarea.md**: Proyectos prácticos

**Información adicional**: [Recursos complementarios](./02_Big_data/Informacion_Adicional.md)

---

## 📋 Estructura del contenido

### Por cada bloque de aprendizaje:

```
Bloque_nombre/
├── Mapa_Conceptual.md      # Diagrama visual de conceptos
├── Teoria.md               # Contenido teórico completo
└── Tarea.md                # Ejercicio práctico con criterios
```

### Por cada módulo:

```
Modulo/
├── 00_Bloque_tematico/     # Contenidos + Actividades
├── 01_Bloque_tematico/     
├── 02_Bloque_tematico/
├── ...
└── Informacion_Adicional.md # Recursos y referencias
```

---

## 🎓 Estructura de aprendizaje

### Recomendación de estudio:

1. **Lectura de teoría** 📖
   - Lee el archivo `Teoria.md` de cada bloque
   - Consulta el `Mapa_Conceptual.md` para visualizar relaciones
   - Complementa con `Informacion_Adicional.md`

2. **Realización de tarea** 💻
   - Implementa los ejercicios del archivo `Tarea.md`
   - Sigue los criterios de evaluación establecidos
   - Documenta resultados y aprendizajes

3. **Integración** 🔗
   - Relaciona conceptos entre bloques
   - Aplica conocimiento a casos reales
   - Participa en proyectos integradores

---

## 🛠️ Tecnologías clave por módulo

### Machine learning - Base
- Python 3.x
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn
- Jupyter Notebook
- CLIPS / Jess (sistemas expertos)

### Machine learning - Avanzado
- **Deep learning**: TensorFlow/Keras, PyTorch
- **Transformers**: Hugging Face, transformers library
- **LLM**: OpenAI API, Anthropic, LLaMA, Mistral
- **Reinforcement learning**: Gym, Stable-Baselines3, Ray RLLib
- **Visualización**: Tensorboard, Weights & Biases

### Big data
- Apache Spark / PySpark
- Apache Hadoop
- Apache Kafka
- SQL (Spark SQL, Presto)
- Cloud: AWS / Azure / Google Cloud
- Herramientas BI: Tableau, Looker, Superset
- Minería: scikit-learn distribuido, Dask

---

## 📊 Mapas de Navegación Rápida

### Ruta principiante (recomendado)
1. Conceptos básicos de IA → ML supervisado → Fundamentos big data
2. Continuación: ML no supervisado → Herramientas procesamiento
3. Intermedio: Validación/evaluación → Análisis masivo
4. Avanzado: Deep learning → Transformers → Minería de datos
5. Especialización: LLM → Reinforcement learning → Sistemas expertos

### Ruta por interés profesional
- **Data scientist**: Conceptos básicos IA → ML supervisado → ML no supervisado → Validación → Deep learning → Minería de datos
- **ML engineer**: Deep learning → Transformers → LLM → Reinforcement learning → Validación → Sistemas expertos
- **Data engineer**: Big data fundamentos → Herramientas Spark → Análisis masivo → Minería de datos

---

## 📈 Requisitos Previos

- Conocimientos básicos de programación (Python)
- Matemática: Álgebra lineal, cálculo, probabilidad
- Conceptos de estructuras de datos
- Familiaridad con línea de comandos

---

## 🔗 Enlaces directos a contenido

### Machine learning
- [Conceptos básicos IA - Bloque 0](./01_Sistemas_aprendizaje_automatico/00_Conceptos_basicos_ia/Teoria.md)
- [ML supervisado - Bloque 1](./01_Sistemas_aprendizaje_automatico/01_Ml_supervisado/Mapa_Conceptual.md)
- [ML no supervisado - Bloque 2](./01_Sistemas_aprendizaje_automatico/02_Ml_no_supervisado/Teoria.md)
- [Validación - Bloque 3](./01_Sistemas_aprendizaje_automatico/03_Validacion_evaluacion/Tarea.md)
- [Deep learning - Bloque 4](./01_Sistemas_aprendizaje_automatico/04_Deep_learning/Teoria.md)
- [Transformers - Bloque 5](./01_Sistemas_aprendizaje_automatico/05_Transformers/Teoria.md)
- [LLM - Bloque 6](./01_Sistemas_aprendizaje_automatico/06_Llm/Teoria.md)
- [Reinforcement learning - Bloque 7](./01_Sistemas_aprendizaje_automatico/07_Reinforcement_learning/Teoria.md)
- [Sistemas expertos - Bloque 8](./01_Sistemas_aprendizaje_automatico/08_Sistemas_expertos/Teoria.md)

### Big data
- [Fundamentos - Bloque 1](./02_Big_data/01_Fundamentos_big_data/Mapa_Conceptual.md)
- [Herramientas - Bloque 2](./02_Big_data/02_Herramientas_procesamiento/Teoria.md)
- [Análisis masivo - Bloque 3](./02_Big_data/03_Analisis_datos_masivos/Tarea.md)
- [Minería datos - Bloque 4](./02_Big_data/04_Mineria_datos/Teoria.md)

---

## ✅ Criterios de calidad y accesibilidad

✓ Contenido estructurado y modular  
✓ Múltiples formatos de presentación  
✓ Ejercicios prácticos progresivos  
✓ Criterios de evaluación claros  
✓ Estándares éticos y legales  
✓ Accesibilidad WCAG 2.1  
✓ Cumplimiento GDPR/CCPA  

---

## 📞 Información de contacto y soporte

Para consultas, sugerencias o reportar errores, contacta con el equipo de desarrollo.

---

## 📄 Licencia

Este material educativo está disponible para uso académico y profesional siguiendo principios de accesibilidad y ética en datos.

---

**Última actualización**: Diciembre 2025  
**Versión**: 3.0 - Estructura optimizada