# Mapa conceptual: machine learning supervisado
## Sistemas de aprendizaje automático - Bloque 1

### Aprendizaje supervisado
Requiere datos etiquetados (entrada-salida conocida)

### Tareas principales
- **Clasificación**: predecir categorías discretas
- **Regresión**: predecir valores continuos

### Algoritmos de clasificación
```
Supervisado
    ├── Clasificación
    │   ├── Árboles de Decisión
    │   ├── Logistic Regression
    │   ├── SVM (Support Vector Machines)
    │   ├── Random Forest
    │   └── Naive Bayes
    └── Regresión
        ├── Regresión Lineal
        ├── Regresión Polinomial
        ├── Ridge/Lasso
        └── SVR (Support Vector Regression)
```

### Flujo de trabajo
```
Datos → Preparación → División Train/Test → 
Entrenamiento → Validación → Evaluación
```

### Métricas de evaluación
- Precisión, Recall, F1-Score (Clasificación)
- MAE, MSE, RMSE (Regresión)
