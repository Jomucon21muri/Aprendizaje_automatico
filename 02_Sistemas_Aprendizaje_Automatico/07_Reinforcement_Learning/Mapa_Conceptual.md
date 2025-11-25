# Mapa Conceptual: Reinforcement Learning
## Sistemas de Aprendizaje Automático - Bloque 7

### Marco MDP (Markov Decision Process)

```
       Estado (S)
           ↓
        Agente
           ↓
      Acción (A)
           ↓
      Ambiente
           ↓
   Recompensa (R) + Nuevo Estado (S')
           ↑
           └─ Iteración
```

### Elementos Clave
- **Agente**: Toma decisiones
- **Ambiente**: Sistema dinámico
- **Estado**: Configuración actual
- **Acción**: Movimiento disponible
- **Recompensa**: Retroalimentación inmediata
- **Política**: Estrategia de decisión
- **Valor**: Utilidad esperada

### Categorías de Algoritmos

```
Reinforcement Learning
    ├── Value-Based
    │   ├── Q-Learning
    │   ├── SARSA
    │   ├── Deep Q-Networks (DQN)
    │   └── Distributional RL
    ├── Policy-Based
    │   ├── Policy Gradient
    │   ├── REINFORCE
    │   ├── Actor-Critic
    │   └── PPO, A3C, TRPO
    ├── Model-Based
    │   ├── Dinámica aprendida
    │   ├── Planning
    │   └── Imagination-augmented
    └── Métodos Híbridos
        ├── A3C
        ├── SAC
        └── TD3
```

### Aplicaciones
- Juegos: Ajedrez, Go, Atari
- Robótica
- Conducción autónoma
- Optimización de recursos
- Finanzas
- Control de sistemas