# Contenido teórico: herramientas de procesamiento
## Big Data - Bloque 2

## 1. Procesamiento por lotes (Batch Processing)

### 1.1 MapReduce
- Paradigma divide y conquista
- Dos fases: Map (parallelizar) y Reduce (combinar)
- Orientado a tolerancia a fallos
- Menos flexible que Spark

### 1.2 Apache Hadoop
- Framework para procesamiento distribuido
- **HDFS**: Sistema de archivos distribuido
- **MapReduce**: Engine de procesamiento
- **YARN**: Gestor de recursos

## 2. Procesamiento de flujo (Stream Processing)

### 2.1 Apache Kafka
- Message broker distribuido
- Topics y particiones
- Tolerancia a fallos
- Usado por: LinkedIn, Netflix, Uber

### 2.2 Apache Flink
- True streaming (no micro-batching)
- Procesamiento estateful
- Window functions
- CEP (Complex Event Processing)

### 2.3 Spark Structured Streaming
- Abstracción de DataFrames
- Compatible con SQL
- Micro-batching internamente

## 3. Apache Spark - análisis profundo

### 3.1 Componentes
- **Spark Core**: Funcionalidad básica RDD
- **Spark SQL**: Query distribuidas
- **Spark MLlib**: Machine Learning
- **Spark GraphX**: Procesamiento de grafos
- **Spark Streaming**: Stream processing

### 3.2 RDD (Resilient Distributed Datasets)
- Colecciones inmutables distribuidas
- Tolerancia a fallos mediante lineage
- Transformaciones lazy: map, filter, join
- Acciones: collect, save, count

### 3.3 DataFrames y Datasets
- Abstracción de datos tabulares
- Optimización automática mediante Catalyst
- SQL queries nativas
- Mejor rendimiento que RDD

### 3.4 Operaciones comunes
- Transformaciones: map, filter, groupBy, join
- Acciones: collect, show, write
- Optimización: caching, partitioning

## 4. Herramientas alternativas

### 4.1 Dask
- Librería Python
- Paralelización de código familiar
- Compatible con pandas, numpy
- Para máquinas single-node con muchos núcleos

### 4.2 Ray
- Framework general computación distribuida
- Mejor que Spark para ciertos workloads
- ML training distribuido
- Rayleigh tuning

### 4.3 Polars
- Procesamiento vectorizado en Rust
- Muy rápido
- API tipo pandas
- Ideal para ETL

## 5. Arquitecturas híbridas

### 5.1 Arquitectura Lambda
- Capa batch: precisión garantizada
- Capa speed: latencia baja
- Capa serving: resultados unificados

### 5.2 Arquitectura Kappa
- Solo capa stream
- Reprocessing de datos históricos
- Menos complejidad que Lambda
