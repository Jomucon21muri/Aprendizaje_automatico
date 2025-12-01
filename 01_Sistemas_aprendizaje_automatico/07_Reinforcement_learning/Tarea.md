# Tarea: reinforcement learning
## Sistemas de aprendizaje automático - Bloque 7

### Objetivo
Implementar y entrenar agentes RL en ambientes de control, de simple a complejo.

### Actividades

#### Actividad 1: Q-Learning en GridWorld
- Ambiente simple: grilla 5x5, objetivo, obstáculos
- Implementa Q-Learning desde cero
- Visualiza tabla Q aprendida
- Analiza convergencia vs iteraciones
- Prueba diferentes learning rates

#### Actividad 2: Deep Q-Network (DQN) en juego simple
- Ambiente: OpenAI Gym (CartPole, MountainCar)
- Implementa DQN con experiencia replay
- Grafica rewards durante entrenamiento
- Compara: sin/con target network
- Experimenta con hiperparámetros

#### Actividad 3: Policy Gradient Methods
- Implementa REINFORCE desde cero
- Ambiente: Gym simple
- Compara varianza vs Actor-Critic
- Análisis: cómo reduce baseline la varianza

#### Actividad 4: Actor-Critic avanzado
- Implementa A3C o PPO
- Ambiente más complejo (Lunar Lander, Atari)
- Paralleliza entrenamiento si es posible
- Compara convergencia con métodos anteriores

#### Actividad 5: exploración de trade-offs
- Crea tabla comparativa:
  - Q-Learning vs Policy Gradient vs Actor-Critic
  - Dimensiones: Sample efficiency, convergencia, complejidad
- Análisis: cuándo usar cada uno

#### Actividad 6: proyecto integrador
Selecciona uno:

**Opción A: juego interactivo**
- Entrena agente en juego Atari o similar
- Crea interfaz para jugar contra agente
- Análisis: estrategias aprendidas

**Opción B: optimización**
- Problema: scheduling, asignación recursos
- Formúlalo como MDP
- Entrena agente RL
- Compara con baseline heurístico

**Opción C: robótica simulada**
- Usa simulador (Gym, MuJoCo, PyBullet)
- Tarea: movimiento coordenado, manipulación
- Entrena con SAC o PPO
- Visualiza comportamiento emergente

### Criterios de evaluación
- Q-Learning básico: 12%
- DQN implementación: 15%
- Policy Gradient: 12%
- Actor-Critic: 15%
- Análisis comparativo: 13%
- Proyecto integrador: 33%

### Entrega
- Notebooks con código comentado
- Gráficos: Rewards, convergencia, análisis
- Ambientes personalizados si aplica
- Agentes entrenados (archivos .pt o .h5)
- Visualización de comportamiento (video o GIF)
- Reporte: Decisiones de diseño, lecciones aprendidas
- Análisis: Limitaciones y mejoras futuras