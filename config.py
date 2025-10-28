#!/usr/bin/env python3
"""
Configuraciones centralizadas del proyecto Untappd Recomender
Todas las constantes y paths del sistema
"""

import os

# Directorio base del proyecto
BASE_DIR = os.path.dirname(__file__)

# Base de datos
DATABASE_FILE = os.path.join(BASE_DIR, "datos", "untappd.db")

# Modelo Two-Tower
MODEL_PATH = os.path.join(BASE_DIR, "datos", "two_tower_model.keras")
MAPPINGS_PATH = os.path.join(BASE_DIR, "datos", "id_mappings.pkl")
USER_MODELS_DIR = os.path.join(BASE_DIR, "datos", "user_models")

# Scripts de entrenamiento (para subprocess)
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_two_tower.py")
FINE_TUNE_SCRIPT = os.path.join(BASE_DIR, "fine_tune_user.py")

# Hiperparámetros del modelo
EMBEDDING_DIM = 32
EPOCHS = 10
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2

# Parámetros de retreinamiento
MIN_EVALUACIONES_PARA_RETRAIN = 10
EVALUACIONES_PARA_FINE_TUNING = 10
EVALUACIONES_PARA_RETRAIN_GLOBAL = 50
DIAS_PARA_RETRAIN_GLOBAL = 1

# Directorios de datos CSV
CSV_DIR = os.path.join(BASE_DIR, "datos", "csv")

# Archivos CSV
BREWERIES_CSV = os.path.join(CSV_DIR, "breweries.csv")
BEERS_CSV = os.path.join(CSV_DIR, "beers.csv")
USERS_CSV = os.path.join(CSV_DIR, "users.csv")
RATINGS_CSV = os.path.join(CSV_DIR, "beer_ratings.csv")

# Configuración de Flask
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Configuración de recomendaciones
DEFAULT_RECOMMENDATIONS = 9
MAX_RECOMMENDATIONS = 20
MIN_SIMILARITY_THRESHOLD = 0.1
MAX_SIMILAR_USERS = 50

# Configuración de admin
ADMIN_USER_ID = "rmarques"
