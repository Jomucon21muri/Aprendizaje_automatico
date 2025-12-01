# Tarea: minería de datos
## Big Data - Bloque 4

### Objetivo
Ejecutar proyecto completo de minería de datos siguiendo metodología CRISP-DM.

### Actividades

#### Actividad 1: Business Understanding y Scoping
- Define problema de negocio claro
- Objetivos específicos (SMART)
- Identificar éxito/fracaso
- Disponibilidad de datos
- Timeline y recursos
- Documento: propuesta de proyecto

#### Actividad 2: Data Understanding y exploración
- Recopila dataset relevante (~100K+ filas)
- EDA exhaustiva:
  - Estadísticas descriptivas
  - Distribuciones
  - Correlaciones
  - Valores faltantes
  - Outliers
- Visualizaciones comprensivas
- Documento: reporte EDA con hallazgos

#### Actividad 3: Data Preparation y Feature Engineering
- Limpieza: Manejo valores faltantes, outliers, inconsistencias
- Integración si múltiples fuentes
- Transformación: Normalización, escalado
- Feature Engineering: Crear 5-10 características nuevas
- Justificación de cada transformación
- Documento: Data preparation report

#### Actividad 4: modelado
- Selecciona 2-3 tareas apropiadas:
  - Clasificación + Regresión
  - Clustering + Clasificación
  - Forecasting + Anomalía
- Implementa múltiples algoritmos (4-6 por tarea)
- Hiperparameter tuning (Grid/Random search)
- Validación cruzada rigurosa
- Comparación de modelos

#### Actividad 5: evaluación y análisis
- Métricas comprensivas (negocio + técnicas)
- ROI estimado del modelo
- Análisis de errores
- Interpretabilidad: SHAP/LIME/Feature Importance
- Fairness check si aplica
- Documentar resultados
- Crear reporte ejecutivo

#### Actividad 6: propuesta de Deployment
- Arquitectura de producción
- API o batch process
- Monitoreo y alertas
- Plan de reentrenamiento
- Consideraciones éticas/legales
- Riesgos y mitigación
- Timeline de implementación

### Criterios de evaluación
- Business Understanding: 10%
- Data Understanding: 15%
- Data Preparation: 20%
- Modeling: 20%
- Evaluation: 20%
- Deployment Planning: 15%

### Entrega
- **Propuesta**: Definición problema (2 págs)
- **Notebook**: Código comentado, end-to-end
- **Reportes**: EDA, Data Prep, Resultados
- **Visualizaciones**: Dashboard ejecutivo
- **Modelos**: Archivos entrenados + objetos preprocessor
- **Documentación**: 
  - Decisiones técnicas
  - Lecciones aprendidas
  - Limitaciones y mejoras
  - Plan de producción
- **Presentación**: Resumen ejecutivo (5 min video o slides)

### Notas
- Enfoque en CRISP-DM completo, no solo accuracy
- Énfasis en preparación de datos
- Explicabilidad es tan importante como precisión
- Considera aspectos éticos y de negocio
- Documentación profesional esperada