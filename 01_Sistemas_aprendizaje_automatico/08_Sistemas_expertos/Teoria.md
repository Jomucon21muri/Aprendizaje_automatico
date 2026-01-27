# Sistemas Expertos: Representación del Conocimiento e Inferencia
## Sistemas de Aprendizaje Automático - Bloque 8

## Resumen

Los Sistemas Expertos constituyen una de las primeras aplicaciones exitosas de la inteligencia artificial, caracterizados por la encapsulación de conocimiento especializado en dominios específicos para replicar el proceso de razonamiento de expertos humanos. Este documento examina los fundamentos teóricos, arquitecturas, metodologías de representación del conocimiento, motores de inferencia y aplicaciones prácticas de estos sistemas que precedieron el auge del aprendizaje automático moderno, pero mantienen relevancia en dominios donde la explicabilidad y el razonamiento basado en reglas son críticos.

## 1. Fundamentos y Contexto Histórico

### 1.1 Definición y Caracterización

Un Sistema Experto se define como un programa informático que contiene conocimiento especializado de un dominio particular y utiliza procesos de inferencia para resolver problemas que normalmente requerirían experiencia humana experta.

**Propiedades Distintivas**:
- **Separación Conocimiento-Control**: Base de conocimiento independiente del motor de inferencia
- **Razonamiento Simbólico**: Manipulación de representaciones explícitas de conocimiento
- **Explicabilidad**: Capacidad de justificar conclusiones mediante trazas de razonamiento
- **Conocimiento Heurístico**: Incorporación de reglas prácticas y experiencia del dominio

**Diferencias con Aprendizaje Automático Moderno**:
- Sistemas Expertos: Conocimiento codificado explícitamente por ingenieros
- Machine Learning: Conocimiento extraído automáticamente de datos
- Sistemas Expertos: Razonamiento deductivo basado en lógica
- Deep Learning: Aprendizaje de representaciones distribuidas subsimbólicas

### 1.2 Evolución Histórica

**Década de 1960-1970: Pioneros**:
- **DENDRAL** (1965): Análisis de espectros de masas en química
- **MYCIN** (1972): Diagnóstico de infecciones bacterianas y prescripción antibiótica
- Demostró viabilidad de sistemas basados en conocimiento

**Década de 1980: Edad de Oro**:
- **R1/XCON** (DEC): Configuración de sistemas informáticos
- **PROSPECTOR**: Exploración geológica
- Boom comercial de sistemas expertos
- Desarrollo de shells y herramientas de construcción

**Década de 1990-2000: Declive y Transformación**:
- Limitaciones de escalabilidad y adquisición de conocimiento
- Surgimiento de enfoques estadísticos y Machine Learning
- Integración con métodos híbridos

**Era Contemporánea**:
- Aplicaciones en dominios regulados (medicina, finanzas, legal)
- Sistemas híbridos: reglas + aprendizaje automático
- XAI (Explainable AI): Renovado interés en interpretabilidad

### 1.3 Ventajas de Sistemas Expertos

**Disponibilidad Perpetua**:
- Conocimiento accesible 24/7
- No sujeto a fatiga, emociones o disponibilidad humana

**Consistencia y Uniformidad**:
- Decisiones reproducibles bajo mismas condiciones
- Eliminación de variabilidad entre expertos diferentes

**Preservación de Conocimiento**:
- Documentación explícita de expertise
- Transferencia de conocimiento organizacional
- Protección contra pérdida de expertos (jubilación, rotación)

**Capacitación y Asistencia**:
- Herramienta educativa para novatos
- Asistencia a decisiones para no-expertos

**Reducción de Costos**:
- Automatización de tareas que requerirían consultas costosas

## 2. Arquitectura de Sistemas Expertos

### 2.1 Base de Conocimiento

Repositorio del conocimiento del dominio en forma estructurada.

**Componentes**:

**Hechos (Facts)**:
- Información estática sobre el dominio
- Ejemplo: "El paciente tiene fiebre de 39°C"
- "París es la capital de Francia"

**Reglas (Rules)**:
- Relaciones causa-efecto, heurísticas
- Formato: SI (condiciones) ENTONCES (conclusiones)
- Ejemplo: 
  ```
  SI temperatura > 38°C AND tos_persistente = Verdadero
  ENTONCES posible_infección_respiratoria
  ```

**Heurísticas**:
- Reglas prácticas basadas en experiencia
- No siempre garantizan solución óptima pero son eficientes
- Ejemplo: "En caso de duda, priorizar seguridad sobre eficiencia"

**Metareglas**:
- Reglas sobre cómo aplicar otras reglas
- Control de estrategia de razonamiento

### 2.2 Motor de Inferencia (Inference Engine)

Mecanismo que manipula conocimiento para derivar conclusiones.

**Encadenamiento Hacia Adelante (Forward Chaining)**:
- **Data-Driven**: Parte de hechos conocidos hacia conclusiones
- **Proceso**:
  1. Evaluar condiciones de todas las reglas
  2. Identificar reglas aplicables (cuyas condiciones se satisfacen)
  3. Ejecutar regla(s) seleccionada(s), añadiendo nuevos hechos
  4. Repetir hasta alcanzar objetivo o no haber reglas aplicables

**Ventajas**:
- Natural para tareas de monitoreo y control
- Genera todas las conclusiones derivables

**Desventajas**:
- Puede derivar hechos irrelevantes
- Menos eficiente para alcanzar objetivo específico

**Encadenamiento Hacia Atrás (Backward Chaining)**:
- **Goal-Driven**: Parte de hipótesis y busca evidencia
- **Proceso**:
  1. Establecer hipótesis/objetivo a probar
  2. Buscar reglas que concluyan ese objetivo
  3. Intentar satisfacer precondiciones (sub-objetivos) de esas reglas
  4. Recursivamente, establecer sub-objetivos como nuevos objetivos
  5. Si se satisfacen todas las condiciones → hipótesis confirmada

**Ventajas**:
- Eficiente para responder consultas específicas
- Evita exploraciones innecesarias

**Desventajas**:
- No descubre hechos no relacionados con objetivo inicial

**Resolución de Conflictos**:
Cuando múltiples reglas son aplicables simultáneamente:
- **Especificidad**: Regla más específica tiene prioridad
- **Recencia**: Favorece reglas que involucran hechos más recientes
- **Prioridad Explícita**: Meta-reglas definen orden

### 2.3 Subsistema de Explicación

Componente crítico para confianza y usabilidad.

**Funcionalidades**:
- **Trazabilidad**: Mostrar cadena de razonamiento
- **Justificación**: Explicar por qué se llegó a conclusión
- **Transparencia**: Exponer reglas utilizadas

**Ejemplo de Traza**:
```
¿Por qué recomendar antibiótico X?
→ Porque regla R1: Infección bacteriana → Antibiótico
→ ¿Por qué infección bacteriana?
  → Porque regla R2: Fiebre alta + Leucocitosis → Infección
  → Hechos observados: Temperatura 39.5°C, Leucocitos 15000/μL
```

**Importancia**:
- Validación por expertos
- Confianza del usuario
- Depuración del sistema
- Cumplimiento regulatorio (medicina, finanzas)

### 2.4 Interfaz de Usuario

Mecanismo de interacción entre sistema y usuario:
- Recopilación de información inicial
- Presentación de conclusiones y recomendaciones
- Consultas de explicación
- Interfaz natural (lenguaje natural, gráfica)

### 2.5 Subsistema de Adquisición de Conocimiento

Herramientas para construir y mantener base de conocimiento:
- Editores de reglas
- Verificadores de consistencia
- Detección de redundancias y conflictos
- Entrevistas estructuradas con expertos
- Aprendizaje automático para refinamiento

## 3. Representación del Conocimiento

### 3.1 Lógica Proposicional

Representación mediante proposiciones y conectivos lógicos:

**Conectivos**:
- ∧ (AND), ∨ (OR), ¬ (NOT), → (IMPLICA), ↔ (IFF)

**Ejemplo**:
```
Paciente_fiebre ∧ Paciente_tos → Posible_gripe
```

**Ventajas**: Simple, decidible
**Limitaciones**: No maneja cuantificadores ni relaciones complejas

### 3.2 Lógica de Predicados (First-Order Logic)

Mayor expresividad mediante predicados, variables, cuantificadores:

**Ejemplo**:
```
∀x (Humano(x) → Mortal(x))
Padre(x, y) ∧ Padre(y, z) → Abuelo(x, z)
```

**Cuantificadores**:
- ∀ (para todo)
- ∃ (existe)

**Ventajas**: Expresividad poderosa
**Desafíos**: Indecidible en general, complejidad computacional

### 3.3 Frames y Redes Semánticas

**Frames**: Estructuras de datos que representan conceptos típicos:
```
Frame: Paciente
  Slots:
    - Nombre: [string]
    - Edad: [integer]
    - Síntomas: [lista]
    - Diagnóstico: [frame Diagnóstico]
```

**Redes Semánticas**: Grafos de nodos (conceptos) y arcos (relaciones)
- Ejemplo: "Ave" IS-A "Animal", "Pingüino" IS-A "Ave"
- Herencia de propiedades

### 3.4 Reglas de Producción

Formato SI-ENTONCES más común en sistemas expertos:

```
REGLA: Diagnóstico_Gripe
SI:
  Temperatura > 38°C AND
  Dolor_muscular = Sí AND
  Fatiga = Sí AND
  NO Congestión_nasal
ENTONCES:
  Diagnóstico = Gripe_probable
  Confianza = 0.8
  Recomendación = Reposo_e_hidratación
```

**Ventajas**:
- Modularidad: Reglas independientes
- Facilidad de modificación
- Natural para expertos

## 4. Incertidumbre y Razonamiento Probabilístico

### 4.1 Factores de Certeza (Certainty Factors)

MYCIN introdujo factores numéricos de confianza:
- CF ∈ [-1, 1]
- CF = 1: Certeza completa (verdadero)
- CF = -1: Certeza completa (falso)
- CF = 0: Desconocimiento total

**Combinación de Evidencias**:
Fórmulas ad-hoc para propagar incertidumbre.

**Limitaciones**: No fundamentado en teoría probabilística rigurosa.

### 4.2 Redes Bayesianas

Representación gráfica de dependencias probabilísticas:
- Nodos: Variables aleatorias
- Arcos: Dependencias probabilísticas (causales)
- Tablas de Probabilidad Condicional (CPTs)

**Inferencia**:
- Cálculo de probabilidades posteriores dado evidencia
- Algoritmos: Variable Elimination, Belief Propagation

**Ventajas**: Fundamento teórico sólido, manejo riguroso de incertidumbre

### 4.3 Lógica Difusa (Fuzzy Logic)

Manejo de vaguedad y conceptos no binarios:
- Membresía gradual en conjuntos: μ_A(x) ∈ [0,1]
- Ejemplo: "Temperatura alta" con transición gradual

**Aplicaciones**: Sistemas de control (AC, lavadoras), toma de decisiones

## 5. Metodología de Desarrollo

### 5.1 Ciclo de Vida

**Fase 1: Identificación del Problema**:
- Determinar viabilidad y adecuación de sistema experto
- Definir alcance y objetivos

**Fase 2: Adquisición del Conocimiento**:
- Entrevistas con expertos del dominio
- Análisis de documentación
- Observación de casos reales
- Cuello de botella principal

**Fase 3: Conceptualización**:
- Estructurar conocimiento adquirido
- Identificar conceptos, relaciones, estrategias de resolución

**Fase 4: Formalización**:
- Representar conocimiento en formalismo elegido
- Diseño de base de conocimiento

**Fase 5: Implementación**:
- Codificación en shell o lenguaje apropiado
- Desarrollo de interfaces

**Fase 6: Validación y Verificación**:
- Testing con casos de prueba
- Validación con expertos
- Refinamiento iterativo

**Fase 7: Mantenimiento**:
- Actualización de conocimiento
- Corrección de errores
- Expansión de capacidades

### 5.2 Herramientas y Shells

**Shells de Sistemas Expertos**:
- **CLIPS**: C Language Integrated Production System
- **Jess**: Java Expert System Shell
- **Drools**: Business Rules Management System
- **Prolog**: Lenguaje de programación lógica

**Características Comunes**:
- Motor de inferencia pre-construido
- Facilidades de representación
- Herramientas de depuración

## 6. Aplicaciones Contemporáneas

### 6.1 Medicina y Salud

- **Diagnóstico Médico**: Asistencia en identificación de enfermedades
- **Prescripción de Tratamientos**: Recomendaciones personalizadas
- **Interpretación de Resultados**: Análisis de laboratorio, imágenes

**Ejemplo**: Sistemas de soporte a decisiones clínicas (CDSS)

### 6.2 Finanzas

- **Evaluación de Riesgos**: Análisis crediticio, detección de fraude
- **Asesoramiento de Inversiones**: Recomendaciones de portafolio
- **Cumplimiento Regulatorio**: Verificación de normativas

### 6.3 Ingeniería y Manufactura

- **Diagnóstico de Fallos**: Identificación de problemas en maquinaria
- **Configuración de Productos**: Sistemas personalizables complejos
- **Planificación y Scheduling**: Optimización de procesos

### 6.4 Legal

- **Análisis de Contratos**: Identificación de cláusulas problemáticas
- **Asesoramiento Legal**: Guía en procedimientos estándar
- **Investigación Jurídica**: Búsqueda de precedentes

### 6.5 Sistemas Híbridos Modernos

Combinación de reglas expertas con Machine Learning:
- **Reglas para Casos Claros**: Decisiones determinísticas
- **ML para Casos Ambiguos**: Aprendizaje de patrones complejos
- **Explicabilidad Mejorada**: Reglas proporcionan interpretabilidad

## 7. Limitaciones y Desafíos

### 7.1 Adquisición de Conocimiento

- **Cuello de Botella**: Proceso manual, costoso, lento
- **Conocimiento Tácito**: Dificultad para explicitar expertise implícita
- **Expertos Múltiples**: Inconsistencias entre expertos

### 7.2 Mantenimiento

- Actualización continua ante cambios en dominio
- Expansión de reglas puede generar inconsistencias
- Testing exhaustivo requerido

### 7.3 Escalabilidad

- Explosión combinatoria en dominios grandes
- Performance degradada con miles de reglas

### 7.4 Conocimiento Incompleto

- Dificultad para manejar situaciones no anticipadas
- Fragilidad ante casos fuera de cobertura

### 7.5 Falta de Aprendizaje

- Sistemas tradicionales no aprenden de experiencia
- Requieren actualización manual

## 8. Futuro y Convergencia con IA Moderna

### 8.1 Sistemas Neuro-Simbólicos

Integración de razonamiento simbólico con redes neuronales:
- Aprendizaje de representaciones + razonamiento lógico
- Combina fortalezas de ambos paradigmas

### 8.2 Explainable AI (XAI)

Renovado interés en explicabilidad:
- Regulaciones (GDPR "derecho a explicación")
- Confianza en sistemas críticos
- Sistemas expertos aportan metodologías de explicación

### 8.3 Knowledge Graphs

Evolución moderna de representación de conocimiento:
- Grafos masivos de entidades y relaciones
- Integración con embeddings neuronales
- Razonamiento y inferencia sobre grafos

## Referencias

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- Jackson, P. (1998). *Introduction to Expert Systems* (3rd ed.). Addison-Wesley.
- Buchanan, B. G., & Shortliffe, E. H. (Eds.). (1984). *Rule-Based Expert Systems*. Addison-Wesley.
- Liebowitz, J. (Ed.). (1998). *The Handbook of Applied Expert Systems*. CRC Press.
