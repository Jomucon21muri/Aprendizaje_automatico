# Mapa conceptual: validación y evaluación de modelos
## Sistemas de aprendizaje automático - Bloque 3

### Conceptos centrales
- **Overfitting**: modelo memoriza datos
- **Underfitting**: modelo muy simple
- **Generalización**: capacidad en datos nuevos

### Técnicas de validación
```
Validación
    ├── Train/Test Split
    ├── K-Fold Cross-Validation
    ├── Stratified K-Fold
    └── Time Series Split
```

### Métricas por tipo

**Clasificación**
- Exactitud (Accuracy)
- Precisión (Precision)
- Recall (Sensibilidad)
- F1-Score
- ROC-AUC

**Regresión**
- MAE, MSE, RMSE
- R² (Coeficiente de Determinación)
- MAPE (Error Porcentual Absoluto Medio)

### Curvas de diagnóstico
- Learning Curves (Entrenamiento vs Validación)
- Validation Curves (Hiperparámetro vs Rendimiento)
- ROC Curves
- Precision-Recall Curves
