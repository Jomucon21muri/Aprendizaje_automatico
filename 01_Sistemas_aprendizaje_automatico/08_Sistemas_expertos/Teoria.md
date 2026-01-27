# Sistemas Expertos: Representación del Conocimiento y Razonamiento Automatizado
## Sistemas de Aprendizaje Automático - Bloque 8

## Resumen

Los Sistemas Expertos representan una rama fundamental de la Inteligencia Artificial clásica, enfocada en emular capacidad de toma de decisiones de expertos humanos en dominios específicos. Este documento explora fundamentos teóricos de representación del conocimiento, motores de inferencia, metodologías de desarrollo, técnicas de adquisición de conocimiento, y arquitecturas de sistemas basados en reglas. Se analiza la evolución desde sistemas rule-based tradicionales hasta enfoques híbridos que integran aprendizaje automático, destacando aplicaciones en diagnóstico médico, sistemas de control industrial, y asesoramiento técnico especializado.

## 1. Fundamentos de Sistemas Expertos

### 1.1 Definición y Características

**Sistema Experto**: Programa computacional que utiliza conocimiento y procedimientos de inferencia para resolver problemas que normalmente requieren experiencia humana especializada.

**Características Fundamentales**:
- **Dominio Específico**: Expertise en área limitada y bien definida
- **Conocimiento Explícito**: Representación formal de expertise humano
- **Razonamiento Simbólico**: Manipulación de símbolos y reglas lógicas
- **Explicabilidad**: Capacidad de justificar conclusiones
- **Separación Conocimiento-Control**: Distinción entre base de conocimiento y motor de inferencia

**Diferencias con Programación Convencional**:
- Conocimiento declarativo vs imperativo
- Flexibilidad: facilidad para actualizar conocimiento
- Manejo de incertidumbre
- Capacidad explicativa

### 1.2 Evolución Histórica

**Orígenes (1960s-1970s)**:
- **DENDRAL** (1965): Primer sistema experto, identificación de estructuras moleculares
- **MYCIN** (1972-1980): Diagnóstico de infecciones bacterianas, factores de certidumbre
- **PROSPECTOR** (1974): Exploración geológica, descubrió depósito de molibdeno valorado en $100M

**Era de Oro (1980s)**:
- Explosión comercial de sistemas expertos
- Desarrollo de shells y herramientas de construcción
- XCON (DEC): Configuración de sistemas computacionales, ahorró $40M anuales
- Lenguajes especializados: LISP, Prolog

**Declive y Renacimiento (1990s-presente)**:
- "AI Winter": Promesas incumplidas, limitaciones en escalabilidad
- Integración con web y bases de datos
- Hibridización con machine learning
- Aplicaciones en sistemas de soporte a decisiones

## 2. Arquitectura de Sistemas Expertos

### 2.1 Componentes Principales

**Base de Conocimiento (Knowledge Base)**:
- Repositorio del conocimiento del dominio
- Hechos: Información factual sobre el mundo
- Reglas: Relaciones entre conceptos del dominio
- Heurísticas: Conocimiento empírico de expertos

**Base de Hechos (Working Memory)**:
- Estado actual del sistema
- Información específica del problema
- Hechos derivados durante inferencia
- Actualizada dinámicamente

**Motor de Inferencia (Inference Engine)**:
- Mecanismo de razonamiento
- Aplica conocimiento a hechos para derivar conclusiones
- Implementa estrategias de búsqueda
- Controla flujo de ejecución

**Interfaz de Usuario**:
- Entrada de datos del problema
- Presentación de resultados
- Explicación de razonamiento
- Interacción con usuarios no-expertos

**Módulo de Explicación**:
- Justifica conclusiones
- Traza cadenas de razonamiento
- Responde "¿por qué?" y "¿cómo?"
- Aumenta confianza del usuario

**Módulo de Adquisición de Conocimiento**:
- Facilita construcción y mantenimiento de KB
- Herramientas para ingenieros del conocimiento
- Verificación de consistencia
- Refinamiento incremental

### 2.2 Flujo de Operación

1. **Entrada**: Usuario proporciona información del problema
2. **Activación**: Reglas cuyos antecedentes coinciden con hechos
3. **Selección**: Estrategia de resolución de conflictos
4. **Ejecución**: Aplicación de regla seleccionada
5. **Actualización**: Modificación de memoria de trabajo
6. **Iteración**: Repetir hasta alcanzar conclusión o agotar reglas
7. **Salida**: Presentación de solución y explicación

## 3. Representación del Conocimiento

### 3.1 Reglas de Producción

**Formato Básico**:
```
SI <condiciones> ENTONCES <acciones>
IF <antecedente> THEN <consecuente>
```

**Ejemplo Médico**:
```
SI paciente tiene fiebre > 38°C
   Y paciente tiene dolor de garganta
   Y paciente tiene ganglios inflamados
ENTONCES posible diagnóstico: faringitis estreptocócica
   CON certeza: 0.8
```

**Ventajas**:
- Modularidad: Reglas independientes
- Naturalidad: Cercano a razonamiento humano
- Flexibilidad: Fácil añadir/modificar reglas
- Transparencia: Comprensible para expertos

**Limitaciones**:
- Mantenimiento complejo en sistemas grandes
- Dificultad para representar conocimiento temporal
- Problemas de eficiencia con muchas reglas
- "Opacity" en interacciones complejas entre reglas

### 3.2 Redes Semánticas

Representación gráfica del conocimiento mediante nodos (conceptos) y arcos (relaciones):

**Características**:
- Nodos: Entidades, objetos, conceptos
- Arcos etiquetados: Relaciones (es-un, parte-de, tiene-propiedad)
- Jerarquías de herencia
- Razonamiento por asociación

**Ejemplo**:
```
Perro --es-un--> Mamífero --es-un--> Animal
Perro --tiene--> 4_patas
Mamífero --tiene--> sangre_caliente
```

Perro hereda propiedades de Mamífero y Animal.

### 3.3 Frames

Estructuras de datos para representar objetos estereotipados:

**Componentes**:
- **Slots**: Atributos o propiedades
- **Facetas**: Características de slots (tipo, rango, valor por defecto, procedimientos)
- **Herencia**: Frames organizados jerárquicamente

**Ejemplo Frame "Automóvil"**:
```
Frame: Automóvil
  Slots:
    - Fabricante: [tipo: string, requerido: sí]
    - Modelo: [tipo: string]
    - Año: [tipo: entero, rango: 1900-2026]
    - Precio: [tipo: real, if-needed: calcular_precio()]
    - Combustible: [default: "gasolina"]
```

**Ventajas**:
- Organización estructurada
- Herencia múltiple
- Procedimientos asociados (if-needed, if-added)
- Valores por defecto

### 3.4 Lógica de Predicados

Formalismo matemático riguroso para representar conocimiento:

**Lógica Proposicional**: Proposiciones verdaderas o falsas, conectivos lógicos (∧, ∨, ¬, →, ↔).

**Lógica de Primer Orden**: Cuantificadores (∀, ∃), predicados, funciones, términos.

**Ejemplo**:
```
∀x (Humano(x) → Mortal(x))
∀x (Mamífero(x) → RespiraPorPulmones(x))
Mamífero(Perro)
```

**Inferencia mediante Resolución**:
- Refutación: Demostrar negación de objetivo lleva a contradicción
- Unificación: Emparejamiento de patrones
- Backward/forward chaining

**Ventajas**:
- Rigor matemático
- Semántica bien definida
- Teoremas de completitud y corrección
- Base para Prolog

## 4. Motores de Inferencia

### 4.1 Forward Chaining (Encadenamiento Hacia Adelante)

**Estrategia Data-Driven**:
- Comienza con hechos conocidos
- Aplica reglas para derivar nuevos hechos
- Progresa desde datos hacia conclusiones

**Algoritmo**:
1. Inicializar memoria de trabajo con hechos iniciales
2. Mientras existan reglas activables:
   - Identificar reglas aplicables (match)
   - Resolver conflictos (seleccionar una regla)
   - Ejecutar regla seleccionada
   - Añadir conclusiones a memoria de trabajo
3. Reportar hechos derivados

**Aplicaciones**:
- Sistemas de monitoreo y control
- Sistemas de alarma
- Procesamiento de eventos
- Cuando objetivo no está predeterminado

**Ventajas**:
- Natural para problemas exploratorios
- Genera todos los hechos derivables
- Útil cuando hay múltiples objetivos

**Desventajas**:
- Puede derivar hechos irrelevantes
- Ineficiente para objetivos específicos

### 4.2 Backward Chaining (Encadenamiento Hacia Atrás)

**Estrategia Goal-Driven**:
- Comienza con objetivo o hipótesis
- Busca reglas que concluyan el objetivo
- Retrocede para verificar antecedentes
- Recursivo hasta validar con hechos

**Algoritmo**:
1. Establecer objetivo inicial
2. Si objetivo es hecho conocido, éxito
3. Si no, buscar reglas cuyo consecuente sea objetivo
4. Para cada regla, establecer antecedentes como subobjetivos
5. Resolver subobjetivos recursivamente
6. Si todos verificados, objetivo demostrado

**Aplicaciones**:
- Sistemas de diagnóstico
- Consulta a bases de conocimiento
- Demostración de teoremas
- Objetivo específico conocido

**Ventajas**:
- Eficiente para objetivos específicos
- Solo deriva hechos relevantes
- Natural para diagnóstico

**Desventajas**:
- Repetición de subproblemas (sin memoization)
- Puede no encontrar soluciones alternativas

### 4.3 Resolución de Conflictos

Cuando múltiples reglas aplicables simultáneamente, estrategias para seleccionar:

**Especificidad**: Preferir reglas más específicas.
**Recency**: Preferir reglas que usen hechos más recientes.
**Prioridad**: Asignación explícita de prioridades a reglas.
**Orden de Reglas**: Secuencia en la base de conocimiento.
**MEA (Means-Ends Analysis)**: Seleccionar regla que reduce diferencia con objetivo.

## 5. Manejo de Incertidumbre

### 5.1 Factores de Certidumbre (MYCIN)

Modelo ad-hoc para razonamiento con incertidumbre:

**Certitude Factor (CF)**: Valor en [-1, 1] representa creencia.
- CF = 1: Certeza absoluta (verdadero)
- CF = -1: Certeza absoluta (falso)
- CF = 0: Desconocido

**Combinación de Evidencias**:
- Evidencias confirman: $CF(h, e_1 \land e_2) = CF(h,e_1) + CF(h,e_2) - CF(h,e_1) \cdot CF(h,e_2)$
- Evidencias refutan: Fórmula simétrica con signo negativo
- Propagación por reglas: $CF(h, e) = CF(h \leftarrow r) \times CF(e)$

**Ventajas**: Simple, intuitivo para expertos.
**Desventajas**: Sin fundamento probabilístico riguroso, problemas con independencia.

### 5.2 Redes Bayesianas

Modelo probabilístico estructurado:

**Componentes**:
- Grafo dirigido acíclico (DAG)
- Nodos: Variables aleatorias
- Arcos: Dependencias probabilísticas
- Tablas de probabilidad condicional (CPT)

**Inferencia**:
$$P(H|E) = \frac{P(E|H)P(H)}{P(E)}$$

**Ventajas sobre CF**:
- Fundamento matemático sólido
- Manejo riguroso de dependencias
- Algoritmos eficientes de inferencia

Sistemas expertos modernos frecuentemente usan redes Bayesianas para incertidumbre.

### 5.3 Lógica Difusa (Fuzzy Logic)

Manejo de imprecisión e información vaga:

**Grados de Pertenencia**: Valor en [0,1] representa grado en que elemento pertenece a conjunto difuso.

**Ejemplo**:
- Temperatura: Frío (< 15°C), Templado (10-25°C), Caliente (> 20°C)
- 18°C: 0.3 Frío, 0.7 Templado

**Operadores**:
- AND: min(A, B)
- OR: max(A, B)
- NOT: 1 - A

**Inferencia Difusa**:
1. Fuzzificación: Convertir entradas crisp a difusas
2. Evaluación de reglas difusas
3. Agregación de conclusiones
4. Defuzzificación: Convertir salida difusa a valor crisp

**Aplicaciones**: Control de procesos, clasificación flexible.

## 6. Desarrollo de Sistemas Expertos

### 6.1 Metodología

**Fases del Ciclo de Vida**:

1. **Identificación del Problema**:
   - Determinar viabilidad
   - Seleccionar dominio apropiado
   - Identificar expertos disponibles
   - Estimar costos y beneficios

2. **Conceptualización**:
   - Definir conceptos clave del dominio
   - Relaciones entre conceptos
   - Tipos de problemas a resolver
   - Estrategias de solución

3. **Formalización**:
   - Especificar estructuras de representación
   - Definir reglas formales
   - Seleccionar esquema de inferencia

4. **Implementación**:
   - Construcción de prototipo
   - Codificación de conocimiento
   - Desarrollo de interfaces

5. **Prueba y Validación**:
   - Casos de prueba
   - Comparación con expertos humanos
   - Refinamiento iterativo

6. **Despliegue y Mantenimiento**:
   - Integración con sistemas existentes
   - Entrenamiento de usuarios
   - Actualización de conocimiento

### 6.2 Adquisición de Conocimiento

**Desafío Principal**: "Knowledge Acquisition Bottleneck" - Dificultad para extraer y formalizar expertise.

**Técnicas**:

**Entrevistas**:
- Estructuradas: Cuestionarios predefinidos
- No estructuradas: Conversación libre
- Semi-estructuradas: Guía flexible

**Observación**:
- Expertos resolviendo problemas reales
- Análisis de think-aloud protocols
- Estudios etnográficos

**Análisis de Casos**:
- Revisión de problemas históricos
- Identificación de patrones
- Abstracción de reglas generales

**Repertory Grid**:
- Técnica psicológica para elicitar constructos personales
- Comparación de triadas de conceptos
- Análisis de similitudes/diferencias

**Automated Knowledge Acquisition**:
- Machine learning desde datos
- Text mining de documentación
- Inducción de reglas

### 6.3 Verificación y Validación

**Verificación**: ¿Sistema construido correctamente? (consistencia interna)
- Reglas contradictorias
- Ciclos infinitos
- Reglas redundantes
- Reglas inalcanzables
- Completitud de casos

**Validación**: ¿Sistema correcto construido? (cumple requisitos)
- Comparación con expertos humanos
- Casos de prueba representativos
- Medidas de rendimiento (precisión, recall, F1)
- Aceptación por usuarios finales

## 7. Shells y Herramientas

**Expert System Shell**: Entorno de desarrollo con motor de inferencia, representación de conocimiento e interfaces, sin conocimiento de dominio específico.

**Ejemplos Clásicos**:
- **CLIPS** (NASA): Lenguaje orientado a reglas, forward chaining
- **JESS** (Java Expert System Shell): Basado en CLIPS para Java
- **Drools**: Business Rules Management System (BRMS) moderno
- **Prolog**: Lenguaje lógico, backward chaining natural

**Funcionalidades Típicas**:
- Editor de reglas
- Depurador y trazador
- Gestor de consultas
- Módulo de explicación
- Integración con bases de datos

## 8. Aplicaciones Contemporáneas

### 8.1 Diagnóstico Médico

- Sistemas de soporte a decisiones clínicas
- Interpretación de análisis de laboratorio
- Diagnóstico de enfermedades raras
- Integración con historiales electrónicos

### 8.2 Configuración y Diseño

- Configuración de productos complejos
- Diseño asistido por computadora
- Planificación de recursos
- Optimización de layouts

### 8.3 Control de Procesos Industriales

- Monitoreo en tiempo real
- Detección de anomalías
- Control adaptativo
- Mantenimiento predictivo

### 8.4 Sistemas de Recomendación

- Asesoramiento financiero
- Recomendaciones de productos
- Planificación de tratamientos
- Tutorías inteligentes

### 8.5 Sistemas Legales

- Análisis de contratos
- Evaluación de cumplimiento regulatorio
- Asistencia en toma de decisiones jurídicas

## 9. Integración con Machine Learning

### 9.1 Sistemas Híbridos

**Motivación**: Combinar fortalezas de SE (explicabilidad, conocimiento experto) con ML (aprendizaje desde datos).

**Enfoques**:

**ML para Construir SE**:
- Inducción de reglas desde datos
- Aprendizaje de árboles de decisión → reglas
- Extracción de reglas desde redes neuronales

**SE para Guiar ML**:
- Incorporación de conocimiento experto como restricciones
- Inicialización de modelos con conocimiento previo
- Interpretación de resultados ML con reglas

**Arquitecturas Colaborativas**:
- SE para razonamiento de alto nivel, ML para tareas específicas
- ML identifica patrones, SE explica y valida
- Sistemas neurosimbólicos

### 9.2 Explicabilidad (XAI)

Con resurgimiento de redes neuronales profundas (black boxes), hay renovado interés en técnicas de SE para:
- Extraer reglas interpretables de modelos complejos
- Proporcionar explicaciones causales
- Aumentar confianza y transparencia
- Cumplir regulaciones (e.g., GDPR derecho a explicación)

## 10. Limitaciones y Desafíos

### 10.1 Limitaciones Tradicionales

- **Cuello de Botella en Adquisición**: Difícil extraer conocimiento tácito
- **Brittleness**: Fragilidad fuera de dominio específico
- **Mantenimiento Costoso**: Actualización de grandes bases de conocimiento
- **Escalabilidad**: Performance degrada con muchas reglas
- **Conocimiento Incompleto**: Dificultad con casos no cubiertos

### 10.2 Desafíos Contemporáneos

- **Integración de Datos Masivos**: Sistemas expertos con big data
- **Conocimiento Dinámico**: Dominios que evolucionan rápidamente
- **Aprendizaje Continuo**: Adaptación automática sin re-ingeniería completa
- **Razonamiento de Sentido Común**: Limitaciones en conocimiento general del mundo
- **Multimodalidad**: Integración de texto, imágenes, sensores

## 11. Futuro de Sistemas Expertos

### 11.1 Tendencias Emergentes

**Neurosymbolic AI**: Arquitecturas que combinan razonamiento simbólico con redes neuronales profundas.

**Knowledge Graphs**: Representaciones semánticas estructuradas a gran escala (Google Knowledge Graph, Wikidata).

**Semantic Web**: Web de datos enlazados, ontologías compartidas, razonamiento automatizado.

**Explainable AI (XAI)**: Demanda creciente de transparencia en sistemas de IA críticos.

### 11.2 Resurgimiento

Aunque eclipsados por ML en décadas recientes, SE experimentan resurgimiento debido a:
- Necesidad de explicabilidad en aplicaciones críticas (medicina, legal, finanzas)
- Limitaciones de modelos puramente basados en datos (requieren grandes datasets, fallan en generalización)
- Complementariedad: Sistemas híbridos aprovechan fortalezas de ambos paradigmas
- Regulaciones que requieren transparencia y audibilidad

## Referencias

- Feigenbaum, E. A. (1977). The art of artificial intelligence: Themes and case studies of knowledge engineering. *IJCAI*.
- Buchanan, B. G., & Shortliffe, E. H. (Eds.). (1984). *Rule-Based Expert Systems: The MYCIN Experiments of the Stanford Heuristic Programming Project*. Addison-Wesley.
- Jackson, P. (1998). *Introduction to Expert Systems* (3rd ed.). Addison-Wesley.
- Giarratano, J. C., & Riley, G. D. (2004). *Expert Systems: Principles and Programming* (4th ed.). PWS Publishing.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. [Capítulos sobre Lógica y Conocimiento]
- Garcez, A., et al. (2019). Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. *Journal of Applied Logics*, 6(4), 611-632.
