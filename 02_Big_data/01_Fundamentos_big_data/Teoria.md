# Fundamentos de Big Data: Infraestructuras y Paradigmas de Procesamiento Masivo
## Big Data - Bloque 1

## Resumen

Big Data representa un cambio de paradigma en cómo organizaciones capturan, almacenan, procesan y analizan datos a escalas sin precedentes. Este documento examina fundamentos teóricos y prácticos de infraestructuras de datos masivos, comenzando con caracterización mediante las "V's" (volumen, velocidad, variedad, veracidad, valor), arquitecturas distribuidas, sistemas de almacenamiento escalables, paradigmas de procesamiento (batch y streaming), y desafíos de diseño de sistemas que manejan petabytes de información. Se proporciona análisis riguroso del ecosistema Hadoop, arquitecturas Lambda y Kappa, y principios de escalabilidad horizontal que fundamentan la revolución de datos contemporánea.

## 1. Definición y Caracterización de Big Data

### 1.1 Las Cinco V's del Big Data

**Volumen**:
- Escala: Terabytes → Petabytes → Exabytes
- Generación masiva continua de datos
- Ejemplos: 500+ horas de video subidos a YouTube por minuto, 6000+ tweets por segundo

**Velocidad**:
- Alta tasa de generación y llegada de datos
- Necesidad de procesamiento en tiempo real o near-real-time
- Streaming: Datos generados continuamente (sensores IoT, transacciones financieras, logs)

**Variedad**:
- **Estructurados**: Bases de datos relacionales, CSV, formato tabular
- **Semi-estructurados**: JSON, XML, logs con formato
- **No estructurados**: Texto libre, imágenes, videos, audio
- Heterogeneidad de fuentes y formatos

**Veracidad**:
- Calidad y confiabilidad de datos
- Datos ruidosos, incompletos, inconsistentes
- Manejo de incertidumbre y errores
- Origen variado con diferentes niveles de confianza

**Valor**:
- Extracción de insights accionables
- ROI de iniciativas de big data
- Transformación de datos raw → información → conocimiento → decisiones

### 1.2 Definición Operacional

**Big Data**: Conjuntos de datos cuyo tamaño, velocidad de generación, o complejidad exceden la capacidad de tecnologías tradicionales de bases de datos y herramientas de análisis para capturar, gestionar y procesar dentro de tiempos tolerables.

**Umbrales Típicos**:
- Volumen > 100GB (histórico, ahora > 1TB)
- Velocidad: Miles de eventos por segundo
- No procesable en sistema único en tiempo razonable

**Implicaciones**:
- Requerimientos de infraestructura distribuida
- Necesidad de paralelización
- Trade-offs en consistencia y disponibilidad

## 2. Motivación y Aplicaciones

### 2.1 Fuentes Generadoras de Big Data

**Redes Sociales**:
- Twitter: 500M+ tweets/día
- Facebook: 4 petabytes datos nuevos/día
- Instagram: 95M fotos/día

**Internet of Things (IoT)**:
- Sensores industriales
- Wearables de salud
- Smart cities
- Vehículos conectados: Tesla genera ~10GB por vehículo/día

**Transacciones Comerciales**:
- E-commerce: Clics, compras, comportamiento navegación
- Sistemas financieros: Trading de alta frecuencia, transacciones bancarias

**Científicas**:
- Large Hadron Collider: 30 petabytes/año
- Genómica: Secuenciación genoma humano genera ~200GB
- Astronomía: Square Kilometre Array generará 700TB/segundo

**Logs y Telemetría**:
- Servidores web
- Aplicaciones móviles
- Infraestructuras de nube

### 2.2 Casos de Uso Empresariales

**Personalización y Recomendaciones**:
- Netflix: Sistema de recomendación sobre 100M+ suscriptores
- Amazon: Motor de recomendación procesa billones de interacciones

**Optimización Operacional**:
- UPS: Optimización de rutas ahorra 10M galones combustible/año
- Walmart: Análisis de 2.5 petabytes transacciones/hora para optimizar inventarios

**Detección de Fraude**:
- Tarjetas de crédito: Análisis en tiempo real de transacciones
- Seguros: Detección de patrones sospechosos en claims

**Mantenimiento Predictivo**:
- GE: Análisis de sensores en turbinas para predecir fallos
- Manufactura: Reducción de downtime no planificado

**Análisis de Sentimiento**:
- Marcas monitoreando redes sociales
- Análisis de opinión de clientes a escala

## 3. Desafíos Técnicos

### 3.1 Almacenamiento

**Escalabilidad Vertical vs Horizontal**:
- **Vertical (Scale-up)**: Añadir recursos a máquina única (CPU, RAM, disco)
  - Límites físicos
  - Costoso
  - Single point of failure
  
- **Horizontal (Scale-out)**: Añadir más máquinas al cluster
  - Virtualmente ilimitado
  - Commodity hardware
  - Mejor relación coste-rendimiento
  - Complejidad en coordinación

**Sistemas de Archivos Distribuidos**:
- Replicación para tolerancia a fallos
- Particionamiento (sharding)
- Consistencia distribuida
- Localidad de datos

**Cost-Effectiveness**:
- Almacenamiento en disco vs SSD vs memoria
- Compresión de datos
- Tiering: Hot vs Cold storage

### 3.2 Procesamiento

**Paralelización**:
- Descomposición de tareas en subtareas independientes
- Distribución de cómputo a través de nodos
- Agregación de resultados parciales

**Localidad de Datos**:
- "Move computation to data, not data to computation"
- Minimizar transferencia de datos en red
- Co-locación de cómputo y almacenamiento

**Tolerancia a Fallos**:
- Inevitabilidad de fallos en clusters grandes
- Checkpointing y recomputación
- Replicación de tareas especulativas
- Recuperación automática

**Consistencia**:
- Trade-off con disponibilidad y tolerancia a particiones (CAP Theorem)
- Consistencia eventual en muchos sistemas big data

### 3.3 Calidad de Datos

**Limpieza y Preprocesamiento**:
- 80% del tiempo en proyectos de datos
- Valores faltantes, outliers, duplicados
- Normalización y estandarización

**Integración de Fuentes Heterogéneas**:
- Schema mapping
- Resolución de entidades
- Fusión de datos

**Privacidad y Seguridad**:
- Anonimización y privacidad diferencial
- Encriptación en reposo y en tránsito
- Control de acceso granular
- Cumplimiento regulatorio (GDPR, HIPAA)

## 4. Arquitecturas de Big Data

### 4.1 Arquitectura Lambda

**Propuesta**: Nathan Marz (2011)

**Capas**:

**1. Batch Layer**:
- Almacena dataset maestro inmutable
- Precomputa batch views mediante procesamiento exhaustivo
- Alta latencia (horas) pero alta precisión
- Tecnologías: Hadoop MapReduce, Spark Batch

**2. Speed Layer**:
- Procesa datos en tiempo real
- Compensa latencia de batch layer
- Vistas incrementales de baja latencia
- Tecnologías: Storm, Flink, Spark Streaming

**3. Serving Layer**:
- Indexa batch y speed views para queries rápidas
- Merge de resultados batch y real-time
- Tecnologías: ElasticSearch, Druid, Cassandra

**Ecuación**:
$$\text{Query}(t) = \text{BatchView}(t_0 \rightarrow t - \Delta) + \text{SpeedView}(t - \Delta \rightarrow t)$$

donde $\Delta$ es latencia del batch layer.

**Ventajas**:
- Tolerancia a fallos humanos (datos inmutables)
- Precisión eventual mediante recomputación batch
- Queries de baja latencia vía speed layer

**Desventajas**:
- Complejidad: Mantener dos pipelines separados
- Lógica duplicada en batch y speed layers
- Costes operacionales altos

### 4.2 Arquitectura Kappa

**Propuesta**: Jay Kreps (LinkedIn, 2014)

**Simplificación**: Eliminar batch layer, usar solo streaming con reprocessing.

**Componentes**:
1. **Stream Processing**: Única pipeline de procesamiento
2. **Log Distribuido**: Fuente de verdad (e.g., Kafka)
3. **Reprocessing**: Cuando cambios en lógica, reproducir log desde inicio

**Ventajas**:
- Simplicidad: Solo una codebase
- Menor overhead operacional
- Todo es streaming

**Desventajas**:
- Reprocessing de datos históricos puede ser costoso
- Requiere stream processor maduro
- No siempre apropiado para analíticas batch complejas

**Cuándo Usar**:
- Pipeline principalmente real-time
- Lógica de procesamiento relativamente simple
- Cambios de código infrecuentes

### 4.3 Comparación Lambda vs Kappa

| Aspecto | Lambda | Kappa |
|---------|--------|-------|
| Complejidad | Alta (2 pipelines) | Baja (1 pipeline) |
| Latencia | Baja (speed) + Batch | Baja |
| Precisión | Alta (batch) | Depende del stream processor |
| Operación | Compleja | Más simple |
| Reprocessing | Batch natural | Reproducir log |
| Uso típico | Analíticas complejas | Procesamiento real-time |

## 5. Ecosistema Hadoop

### 5.1 Hadoop Core

**HDFS (Hadoop Distributed File System)**:
- Sistema de archivos distribuido y tolerante a fallos
- Diseñado para almacenar archivos muy grandes (GB-TB)
- Write-once, read-many
- Bloques típicos: 128MB o 256MB
- Replicación (factor típico: 3)

**Arquitectura HDFS**:
- **NameNode**: Maestro que mantiene metadata (namespace, mapping bloques a DataNodes)
- **DataNodes**: Esclavos que almacenan bloques reales
- **Secondary NameNode**: Checkpointing de metadata (no es standby)

**Operaciones**:
- `put`: Subir archivo a HDFS
- `get`: Descargar de HDFS
- `ls`: Listar directorios
- `rm`: Eliminar archivos

**Tolerancia a Fallos**:
- Replicación de bloques en múltiples DataNodes (diferentes racks)
- Detección automática de fallos (heartbeats)
- Re-replicación de bloques bajo-replicados

**MapReduce (V1)**:
- Paradigma de programación para procesamiento paralelo
- **Map**: Transformar cada elemento input → (key, value) pairs
- **Shuffle & Sort**: Agrupar valores por clave
- **Reduce**: Agregar valores por clave → output

**Ejemplo Word Count**:
```
Map(documento):
  Para cada palabra w en documento:
    Emit(w, 1)

Reduce(palabra, counts):
  suma = 0
  Para cada count en counts:
    suma += count
  Emit(palabra, suma)
```

**Componentes MapReduce V1**:
- **JobTracker**: Maestro (scheduling, monitoring)
- **TaskTrackers**: Esclavos (ejecutan map/reduce tasks)

**Limitaciones MapReduce**:
- Alta latencia (diseñado para batch)
- I/O a disco entre stages
- Solo modelo map-reduce (no DAGs complejos)
- JobTracker como bottleneck

### 5.2 YARN (Yet Another Resource Negotiator)

**Motivación**: Superar limitaciones de MapReduce V1, separar gestión de recursos de framework de programación.

**Arquitectura**:
- **ResourceManager (RM)**: Maestro global de recursos
  - **Scheduler**: Asigna recursos a aplicaciones
  - **ApplicationsManager**: Acepta jobs, negocia primer contenedor
  
- **NodeManager (NM)**: Agente por nodo, gestiona contenedores
  
- **ApplicationMaster (AM)**: Por aplicación, negocia recursos con RM, coordina ejecución

**Flujo de Ejecución**:
1. Cliente submit aplicación a RM
2. RM aloca contenedor para ApplicationMaster
3. AM se registra con RM
4. AM solicita contenedores para tareas
5. RM concede recursos
6. AM lanza tareas en NodeManagers
7. Tareas reportan progreso a AM
8. AM reporta a RM y cliente
9. AM se desregistra al completar

**Ventajas**:
- Multi-tenancy: Múltiples frameworks sobre YARN (Spark, Flink, Tez)
- Mejor utilización de recursos
- Escalabilidad: RM sin lógica de aplicación específica

### 5.3 Ecosistema Hadoop Extendido

**Almacenamiento y Formato**:
- **HBase**: Base de datos NoSQL distribuida sobre HDFS (inspirada en Google Bigtable)
- **Parquet**: Formato columnar optimizado para analíticas
- **Avro**: Sistema de serialización con schemas evolutivos
- **ORC**: Formato columnar optimizado (Hive)

**Procesamiento**:
- **Hive**: Data warehouse, queries SQL sobre HDFS
- **Pig**: Lenguaje de flujo de datos de alto nivel (Pig Latin)
- **Spark**: Motor de procesamiento in-memory, 100x más rápido que MapReduce
- **Tez**: Motor de ejecución DAG para Hive y Pig

**Streaming**:
- **Storm**: Procesamiento de streams real-time
- **Flink**: Procesamiento unificado batch y streaming
- **Kafka**: Plataforma de streaming distribuida

**Coordinación y Gestión**:
- **ZooKeeper**: Servicio de coordinación distribuida (consensus, configuration, locks)
- **Oozie**: Scheduler de workflows
- **Ambari**: Provisioning, gestión y monitoreo de clusters Hadoop

**Ingesta**:
- **Flume**: Ingesta de logs y eventos
- **Sqoop**: Transferencia bulk entre Hadoop y RDBMS
- **Kafka Connect**: Conectores para Kafka

## 6. Teorema CAP y Consistencia

### 6.1 Teorema CAP (Brewer's Theorem)

**Enunciado**: En un sistema distribuido, es imposible garantizar simultáneamente:
- **C (Consistency)**: Todos los nodos ven los mismos datos al mismo tiempo
- **A (Availability)**: Toda petición recibe respuesta (sin garantía de datos más recientes)
- **P (Partition Tolerance)**: Sistema continúa operando a pesar de particiones de red

**Implicación**: Solo puedes elegir 2 de 3 propiedades.

**Particiones de Red Son Inevitables** en sistemas distribuidos, así que en práctica el trade-off es **C vs A** durante particiones.

**Clasificación de Sistemas**:
- **CP (Consistency + Partition Tolerance)**: HBase, MongoDB (strong consistency), Zookeeper
  - Durante partición, sistema puede volverse indisponible
  
- **AP (Availability + Partition Tolerance)**: Cassandra, DynamoDB, CouchDB
  - Sistema siempre disponible pero puede retornar datos stale
  
- **CA (Consistency + Availability)**: RDBMS tradicionales (MySQL, PostgreSQL)
  - En teoría, pero P es inevitable en práctica → sistemas distribuidos son CP o AP

### 6.2 Modelos de Consistencia

**Strong Consistency (Linearizability)**:
- Toda lectura ve escritura más reciente
- Equivalente a sistema con única copia
- Alto costo en latencia y disponibilidad

**Eventual Consistency**:
- Si no hay nuevas escrituras, eventualmente todas las réplicas convergirán
- Lecturas pueden retornar valores stale temporalmente
- Alta disponibilidad y baja latencia

**Causal Consistency**:
- Operaciones causalmente relacionadas vistas en mismo orden
- Operaciones concurrentes pueden verse en ordenes diferentes

**Read-Your-Writes Consistency**:
- Proceso que escribe ve su propia escritura en lecturas subsecuentes

**Monotonic Reads**:
- Si proceso lee valor de objeto, lecturas subsecuentes no ven versiones más antiguas

## 7. Principios de Diseño de Sistemas Big Data

### 7.1 Particionamiento (Sharding)

**Objetivo**: Distribuir datos a través de múltiples nodos para balance de carga y paralelismo.

**Estrategias**:

**Hash Partitioning**:
$$\text{partition} = \text{hash}(\text{key}) \mod N$$

- Distribución uniforme
- No preserva rango
- Dificulta queries de rango

**Range Partitioning**:
- Dividir por rangos de claves: [0-999], [1000-1999], ...
- Queries de rango eficientes
- Riesgo de hotspots si accesos no uniformes

**Consistent Hashing**:
- Minimiza redistribución cuando nodos añadidos/removidos
- Usado en Cassandra, DynamoDB

**Desafíos**:
- Hotspots: Algunas particiones más accedidas que otras
- Skew: Particiones desbalanceadas en tamaño
- Cross-partition queries: Requieren scatter-gather

### 7.2 Replicación

**Objetivos**:
- **Tolerancia a fallos**: Disponibilidad ante fallos de nodos
- **Escalabilidad de lecturas**: Servir lecturas desde múltiples réplicas
- **Localidad geográfica**: Réplicas cercanas a usuarios

**Estrategias de Replicación**:

**Master-Slave (Primary-Replica)**:
- Una réplica primaria maneja escrituras
- Réplicas secundarias replican cambios
- Lecturas desde secundarias
- Ejemplo: MySQL replication, MongoDB (default)

**Multi-Master**:
- Múltiples réplicas aceptan escrituras
- Resolución de conflictos necesaria
- Mayor complejidad
- Ejemplo: Cassandra, DynamoDB

**Quorum-Based**:
- Escritura exitosa si $W$ réplicas confirman
- Lectura de $R$ réplicas
- Consistencia fuerte si $W + R > N$ (N = total réplicas)
- Trade-off: Consistencia vs latencia vs disponibilidad

### 7.3 Compresión y Codificación

**Importancia**:
- Reducir almacenamiento (costes)
- Reducir I/O (bottleneck común)
- Reducir transferencia de red

**Técnicas**:
- **Compresión**: Snappy (rápida, baja ratio), Gzip (lenta, alta ratio), LZ4, Zstandard
- **Codificación columnar**: Parquet, ORC - compresión superior para analíticas
- **Delta encoding**: Almacenar diferencias
- **Run-length encoding**: Eficiente para datos repetidos
- **Dictionary encoding**: Mapear valores repetidos a IDs pequeños

**Trade-offs**:
- CPU para compress/decompress vs I/O ahorrado
- Ratio de compresión vs velocidad

## 8. Procesamiento Batch vs Streaming

### 8.1 Procesamiento Batch

**Características**:
- Procesamiento de datos históricos acumulados
- Alta latencia (horas, días)
- Alta throughput
- Tareas complejas, iterativas

**Casos de Uso**:
- ETL diarios/semanales
- Entrenamiento de modelos ML
- Generación de reportes agregados
- Análisis históricos exhaustivos

**Tecnologías**: Hadoop MapReduce, Spark Batch, Hive, Pig

### 8.2 Procesamiento Streaming

**Características**:
- Procesamiento de datos en tiempo real conforme llegan
- Baja latencia (milisegundos, segundos)
- Datos potencialmente infinitos
- Operaciones incrementales

**Modelos**:

**Record-at-a-time**:
- Procesar cada evento individual
- Storm (tuples), Kinesis

**Micro-batching**:
- Agrupar eventos en mini-batches (segundos)
- Spark Structured Streaming

**Hybrid**:
- Flink (streaming nativo con opciones batch)

**Desafíos Streaming**:
- **Windowing**: Cómo agrupar eventos temporalmente
  - Tumbling windows (sin superposición)
  - Sliding windows (con superposición)
  - Session windows (basadas en inactividad)
  
- **Event Time vs Processing Time**:
  - Event time: Timestamp cuando evento ocurrió
  - Processing time: Cuando sistema procesa evento
  - Out-of-order events, late arrivals
  
- **State Management**:
  - Mantener estado agregado eficientemente
  - Checkpointing para tolerancia a fallos
  
- **Exactly-Once Semantics**:
  - Garantizar cada evento procesado exactamente una vez
  - Complejo en sistemas distribuidos

**Tecnologías**: Apache Flink, Spark Structured Streaming, Apache Storm, Kafka Streams, Google Dataflow

### 8.3 Unified Processing

**Tendencia**: Unificar batch y streaming en un solo framework.

**Apache Flink**:
- Streaming como caso general, batch como streaming con límites
- Event time nativo
- Exactly-once state consistency

**Spark Structured Streaming**:
- API unificada para batch y streaming
- Modelo incremental sobre Spark SQL
- Micro-batching

## 9. Tendencias Contemporáneas

### 9.1 Cloud-Native Big Data

**Desacoplamiento Compute-Storage**:
- Almacenamiento en object stores (S3, Azure Blob, GCS)
- Compute efímero y elástico
- Ventajas: Escalabilidad independiente, reducción costes

**Servicios Gestionados**:
- AWS: EMR, Athena, Redshift, Kinesis
- GCP: Dataproc, BigQuery, Dataflow
- Azure: HDInsight, Synapse Analytics

### 9.2 DataOps y Data Mesh

**DataOps**:
- DevOps aplicado a datos
- CI/CD para pipelines de datos
- Monitoreo y observabilidad
- Data quality tests

**Data Mesh**:
- Arquitectura descentralizada
- Domain-oriented data ownership
- Data as a product
- Self-serve data platform

### 9.3 Lakehouse Architecture

**Motivación**: Combinar flexibilidad de data lakes con performance y governance de data warehouses.

**Componentes**:
- **Storage**: Data lake (Parquet, Delta Lake, Iceberg)
- **Metadata Layer**: Transacciones ACID, schema evolution
- **Query Engines**: Spark, Presto, Dremio

**Delta Lake, Apache Iceberg, Apache Hudi**: Formatos que proveen ACID sobre data lakes.

## Referencias

- Dean, J., & Ghemawat, S. (2004). MapReduce: Simplified data processing on large clusters. *OSDI*.
- Shvachko, K., et al. (2010). The Hadoop Distributed File System. *MSST*.
- Marz, N., & Warren, J. (2015). *Big Data: Principles and best practices of scalable real-time data systems*. Manning.
- Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly.
- Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. *CACM*, 59(11).
- Kreps, J., Narkhede, N., & Rao, J. (2011). Kafka: A distributed messaging system for log processing. *NetDB*.
- Brewer, E. A. (2012). CAP twelve years later: How the "rules" have changed. *IEEE Computer*, 45(2).
- Vavilapalli, V. K., et al. (2013). Apache Hadoop YARN: Yet another resource negotiator. *SOCC*.
