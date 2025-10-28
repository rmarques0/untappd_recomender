# Untappd Recomender

Sistema de recomendación de cervezas basado en datos de Untappd, implementando múltiples estrategias de recomendación incluyendo filtrado colaborativo, filtrado por contenido y modelo neural Two-Tower.

## Arquitectura

### Estrategias de Recomendación

- **Popularidad**: Recomendaciones basadas en cervezas más populares
- **Filtrado Colaborativo**: Basado en usuarios similares
- **Two-Tower Neural**: Modelo de deep learning con embeddings
- **Híbrida**: Estrategia adaptativa basada en disponibilidad de datos del usuario

### Sistema de Retreinamiento Incremental

- **Retreinamiento Global**: Cuando usuario nuevo alcanza 10+ evaluaciones
- **Fine-tuning Individual**: Cada 10 evaluaciones adicionales del usuario
- **Retreinamiento Periódico**: Cada 50 evaluaciones acumuladas o diariamente

## Estructura del Proyecto

```
untappd_recomender/
├── app.py              # Flask app principal
├── recomendar.py       # Lógica de recomendación
├── config.py           # Configuraciones centralizadas
├── database.py         # Gestión de base de datos
├── models.py           # Entrenamiento y retreinamiento
├── utils.py            # Utilidades y métricas
├── requirements.txt    # Dependencias Python
├── templates/          # Templates HTML
│   ├── admin.html
│   ├── cerveza_detalle.html
│   ├── recomendaciones.html
│   └── ...
└── datos/              # Datos y modelos
    ├── untappd.db      # Base de datos SQLite
    ├── two_tower_model.keras
    ├── id_mappings.pkl
    └── csv/            # Datos CSV originales
```

## Instalación

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd untappd_recomender
```

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar Base de Datos

```bash
# Migrar CSVs a SQLite
python -c "from database import main; main()"
```

### 4. Entrenar Modelo Two-Tower

```bash
# Entrenar modelo inicial
python -c "from models import train_model; train_model()"
```

### 5. Ejecutar Aplicación

```bash
python app.py
```

## Datos

### Fuentes de Datos

- **Usuarios**: 46,795 usuarios únicos
- **Cervezas**: 12,420 cervezas catalogadas
- **Cervecerías**: 1,000 cervecerías
- **Ratings Históricos**: 201,226 evaluaciones

### Esquema de Base de Datos

- `usuarios`: Información de usuarios
- `cervezas`: Catálogo de cervezas con features (ABV, IBU, estilo)
- `cervecerias`: Información de cervecerías
- `interaccion`: Interacciones del sistema de recomendación
- `ratings_historicos`: Ratings históricos del Untappd
- `model_updates`: Control de retreinamiento por usuario
- `retrain_history`: Histórico de retreinamientos globales
- `retrain_config`: Configuración de retreinamiento

## Modelo Two-Tower

### Arquitectura

- **User Tower**: Embeddings de usuarios (32 dimensiones)
- **Beer Tower**: Embeddings de cervezas (32 dimensiones)
- **Style Tower**: Embeddings de estilos (16 dimensiones)
- **Brewery Tower**: Embeddings de cervecerías (16 dimensiones)
- **Features Numéricas**: ABV e IBU normalizados
- **Ranking Network**: Red densa para predicción final

### Features Utilizadas

- **Categóricas**: user_id, beer_id, style, brewery_id
- **Numéricas**: ABV (0-20%), IBU (0-100)
- **Target**: Rating normalizado (0-1)

## Sistema de Retreinamiento

### Triggers Automáticos

1. **Usuario Nuevo**: 10+ evaluaciones → Retreinamiento global
2. **Fine-tuning**: Cada 10 evaluaciones adicionales
3. **Retreinamiento Periódico**: 50 evaluaciones acumuladas O 1 día

### Ejecución en Background

- Retreinamientos ejecutan via `subprocess`
- No bloquean la interfaz de usuario
- Registro completo en base de datos

### Control Manual

- Botón de retreinamiento manual en panel admin
- Acceso restringido a usuario admin

## Uso del Sistema

### Interfaz Web

1. **Recomendaciones**: Página principal con cervezas recomendadas
2. **Detalle de Cerveza**: Información detallada y evaluación
3. **Historial**: Evaluaciones previas del usuario
4. **Admin**: Panel de administración (solo admin)

### API de Recomendación

```python
import recomendar

# Recomendación adaptativa
recomendaciones = recomendar.recomendar(user_id, N=9)

# Recomendación específica Two-Tower
recomendaciones = recomendar.recomendar_two_tower(user_id, N=9)
```

## Métricas y Análisis

### Métricas de Recomendación

- **NDCG**: Normalized Discounted Cumulative Gain
- **Precision@K**: Precisión en top-K recomendaciones
- **Recall@K**: Recall en top-K recomendaciones
- **F1@K**: F1-score en top-K recomendaciones

### Análisis de Datos

```bash
# Resumen rápido
python -c "from utils import resumen_rapido; resumen_rapido()"

# Análisis completo
python -c "from utils import main; main()"
```

## Configuración

### Variables Principales (`config.py`)

- `DATABASE_FILE`: Ruta de base de datos
- `MODEL_PATH`: Ruta del modelo Two-Tower
- `EMBEDDING_DIM`: Dimensión de embeddings (32)
- `EPOCHS`: Épocas de entrenamiento (10)
- `BATCH_SIZE`: Tamaño de batch (256)
- `ADMIN_USER_ID`: Usuario administrador

### Parámetros de Retreinamiento

- `MIN_EVALUACIONES_PARA_RETRAIN`: 10
- `EVALUACIONES_PARA_FINE_TUNING`: 10
- `EVALUACIONES_PARA_RETRAIN_GLOBAL`: 50
- `DIAS_PARA_RETRAIN_GLOBAL`: 1

## Mantenimiento

### Entrenamiento Manual

```bash
# Entrenar modelo completo
python -c "from models import train_model; train_model()"

# Fine-tuning específico
python -c "from models import fine_tune_user_model; fine_tune_user_model('user_id')"
```

### Migración de Datos

```bash
# Migrar CSVs a SQLite
python -c "from database import main; main()"
```

### Análisis de Sistema

```bash
# Verificar estado del sistema
python -c "from models import contar_evaluaciones_usuario; print(contar_evaluaciones_usuario('user_id'))"
```

## Deploy en PythonAnywhere

### Archivos Necesarios

- `app.py`
- `recomendar.py`
- `config.py`
- `database.py`
- `models.py`
- `utils.py`
- `requirements.txt`
- `templates/` (carpeta completa)
- `datos/untappd.db`
- `datos/two_tower_model.keras`
- `datos/id_mappings.pkl`

### Configuración en PythonAnywhere

1. Subir archivos via Files tab
2. Instalar dependencias via Bash console
3. Configurar WSGI file para Flask
4. Reiniciar web app

## Logs y Monitoreo

### Logs de Retreinamiento

- Histórico completo en tabla `retrain_history`
- Estados: running, completed, error
- Timestamps de inicio y fin

### Métricas de Performance

- Tiempo de entrenamiento
- Métricas de modelo (Loss, MAE)
- Conteos de datos procesados

## Contribución

### Estructura de Código

- **Comentarios**: En español
- **Código**: En inglés
- **Docstrings**: En español
- **Variables**: En inglés

### Convenciones

- Funciones: `snake_case`
- Clases: `PascalCase`
- Constantes: `UPPER_CASE`
- Archivos: `snake_case.py`

## Licencia

Este proyecto es parte del curso de Sistemas de Recomendación de la Universidad de Buenos Aires.

## Autores

- **Estudiante**: [Nombre del estudiante]
- **Curso**: Sistemas de Recomendación
- **Universidad**: Universidad de Buenos Aires
- **Año**: 2025
