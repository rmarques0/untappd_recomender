#!/usr/bin/env python3
"""
Modelos de Machine Learning y Retreinamiento
Consolidado desde train_two_tower.py, fine_tune_user.py, retrain_manager.py
"""

import sqlite3
import subprocess
import os
import sys
import pickle
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime, timedelta

from config import (
    DATABASE_FILE, MODEL_PATH, MAPPINGS_PATH, USER_MODELS_DIR,
    EMBEDDING_DIM, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT,
    MIN_EVALUACIONES_PARA_RETRAIN, EVALUACIONES_PARA_FINE_TUNING,
    EVALUACIONES_PARA_RETRAIN_GLOBAL, DIAS_PARA_RETRAIN_GLOBAL
)
from database import get_db_connection

# =============================================================================
# SECCIÓN: ENTRENAMIENTO DEL MODELO TWO-TOWER
# =============================================================================

def preparar_datos():
    """
    Carga y prepara datos para entrenamiento
    
    Returns:
        tuple: (user_ids, beer_ids, style_ids, brewery_ids, abv_normalized, ibu_normalized, ratings, mappings)
    """
    print("📊 Cargando datos desde SQLite...")
    
    # Conectar a base de datos
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Cargar ratings históricos con features adicionales
    cursor.execute("""
        SELECT r.username as user_id, r.beer_id, r.rating,
               c.style, c.brewery_id, c.abv, c.ibu
        FROM ratings_historicos r
        JOIN cervezas c ON r.beer_id = c.beer_id
        WHERE r.rating > 0
    """)
    data = cursor.fetchall()
    
    # Cargar TODAS las cervezas disponibles para poder recomendar
    cursor.execute("SELECT DISTINCT beer_id FROM cervezas")
    all_beers = [row['beer_id'] for row in cursor.fetchall()]
    
    conn.close()
    
    if not data:
        raise Exception("No hay datos de interacciones en la base de datos")
    
    print(f"   ✓ {len(data)} interacciones cargadas")
    
    # Extraer usuarios únicos de las interacciones
    users = sorted(set(row['user_id'] for row in data))
    # Usar TODAS las cervezas disponibles, no solo las que tienen ratings
    beers = sorted(all_beers)
    
    # Extraer features categóricas únicas
    styles = sorted(set(row['style'] for row in data if row['style']))
    breweries = sorted(set(row['brewery_id'] for row in data if row['brewery_id']))
    
    print(f"   ✓ {len(users)} usuarios únicos")
    print(f"   ✓ {len(beers)} cervezas únicas")
    print(f"   ✓ {len(styles)} estilos únicos")
    print(f"   ✓ {len(breweries)} cervecerías únicas")
    
    # Crear diccionarios de mapeo (ID externo → índice interno)
    user_to_idx = {u: i for i, u in enumerate(users)}
    beer_to_idx = {b: i for i, b in enumerate(beers)}
    style_to_idx = {s: i for i, s in enumerate(styles)}
    brewery_to_idx = {b: i for i, b in enumerate(breweries)}
    
    # Crear diccionarios inversos (índice interno → ID externo)
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_beer = {i: b for b, i in beer_to_idx.items()}
    idx_to_style = {i: s for s, i in style_to_idx.items()}
    idx_to_brewery = {i: b for b, i in brewery_to_idx.items()}
    
    # Preparar arrays para Keras
    user_ids = np.array([user_to_idx[row['user_id']] for row in data], dtype=np.int32)
    beer_ids = np.array([beer_to_idx[row['beer_id']] for row in data], dtype=np.int32)
    ratings = np.array([row['rating'] / 5.0 for row in data], dtype=np.float32)  # Normalizar 0-1
    
    # Features categóricas
    style_ids = np.array([style_to_idx.get(row['style'], 0) for row in data], dtype=np.int32)
    brewery_ids = np.array([brewery_to_idx.get(row['brewery_id'], 0) for row in data], dtype=np.int32)
    
    # Features numéricas (normalizar)
    abv_values = np.array([row['abv'] if row['abv'] else 0.0 for row in data], dtype=np.float32)
    ibu_values = np.array([row['ibu'] if row['ibu'] else 0.0 for row in data], dtype=np.float32)
    
    # Normalizar ABV (0-20%) e IBU (0-100)
    abv_normalized = np.clip(abv_values / 20.0, 0, 1)
    ibu_normalized = np.clip(ibu_values / 100.0, 0, 1)
    
    print(f"   ✓ Datos preparados: {len(user_ids)} interacciones")
    print(f"   ✓ Ratings normalizados a escala 0-1")
    print(f"   ✓ Features categóricas mapeadas")
    print(f"   ✓ Features numéricas normalizadas")
    
    # Crear diccionario con todos los mappings
    mappings = {
        'user_to_idx': user_to_idx,
        'beer_to_idx': beer_to_idx,
        'style_to_idx': style_to_idx,
        'brewery_to_idx': brewery_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_beer': idx_to_beer,
        'idx_to_style': idx_to_style,
        'idx_to_brewery': idx_to_brewery,
        'n_users': len(users),
        'n_beers': len(beers),
        'n_styles': len(styles),
        'n_breweries': len(breweries)
    }
    
    return (user_ids, beer_ids, style_ids, brewery_ids, abv_normalized, ibu_normalized, 
            ratings, mappings)

def crear_modelo(n_users, n_beers, n_styles, n_breweries, embedding_dim=32):
    """
    Crea modelo Two-Tower con múltiples features
    
    Args:
        n_users: Número de usuarios
        n_beers: Número de cervezas
        n_styles: Número de estilos
        n_breweries: Número de cervecerías
        embedding_dim: Dimensión de embeddings
        
    Returns:
        keras.Model: Modelo compilado
    """
    print("\n🏗️  Creando modelo Two-Tower con múltiples features...")
    
    # User Tower
    user_input = keras.layers.Input(shape=[1], name="user_id")
    user_embedding = keras.layers.Embedding(n_users, embedding_dim, name="user_embedding")(user_input)
    user_vec = keras.layers.Flatten()(user_embedding)
    
    # Beer Tower
    beer_input = keras.layers.Input(shape=[1], name="beer_id")
    beer_embedding = keras.layers.Embedding(n_beers, embedding_dim, name="beer_embedding")(beer_input)
    beer_vec = keras.layers.Flatten()(beer_embedding)
    
    # Style Tower
    style_input = keras.layers.Input(shape=[1], name="style_id")
    style_embedding = keras.layers.Embedding(n_styles, 16, name="style_embedding")(style_input)
    style_vec = keras.layers.Flatten()(style_embedding)
    
    # Brewery Tower
    brewery_input = keras.layers.Input(shape=[1], name="brewery_id")
    brewery_embedding = keras.layers.Embedding(n_breweries, 16, name="brewery_embedding")(brewery_input)
    brewery_vec = keras.layers.Flatten()(brewery_embedding)
    
    # Features numéricas
    abv_input = keras.layers.Input(shape=[1], name="abv")
    ibu_input = keras.layers.Input(shape=[1], name="ibu")
    
    # Ranking Network
    concat = keras.layers.Concatenate()([user_vec, beer_vec, style_vec, brewery_vec, abv_input, ibu_input])
    dense1 = keras.layers.Dense(128, activation='relu')(concat)
    dense2 = keras.layers.Dense(64, activation='relu')(dense1)
    output = keras.layers.Dense(1, activation='sigmoid', name="rating")(dense2)
    
    # Crear modelo
    model = keras.Model(
        inputs=[user_input, beer_input, style_input, brewery_input, abv_input, ibu_input], 
        outputs=output
    )
    
    # Compilar
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['mae']
    )
    
    print("   ✓ Arquitectura Two-Tower con múltiples features creada")
    print(f"   ✓ User embeddings: {n_users} usuarios × {embedding_dim} dimensiones")
    print(f"   ✓ Beer embeddings: {n_beers} cervezas × {embedding_dim} dimensiones")
    print(f"   ✓ Style embeddings: {n_styles} estilos × 16 dimensiones")
    print(f"   ✓ Brewery embeddings: {n_breweries} cervecerías × 16 dimensiones")
    print(f"   ✓ Features numéricas: ABV, IBU")
    
    return model

def entrenar_modelo(model, user_ids, beer_ids, style_ids, brewery_ids, abv_values, ibu_values, ratings, epochs=10, batch_size=256):
    """
    Entrena el modelo
    
    Args:
        model: Modelo Keras
        user_ids: Array de user IDs (índices)
        beer_ids: Array de beer IDs (índices)
        style_ids: Array de style IDs (índices)
        brewery_ids: Array de brewery IDs (índices)
        abv_values: Array de valores ABV normalizados
        ibu_values: Array de valores IBU normalizados
        ratings: Array de ratings normalizados
        epochs: Número de épocas
        batch_size: Tamaño de batch
        
    Returns:
        keras.callbacks.History: Historial de entrenamiento
    """
    print(f"\n🎯 Entrenando modelo...")
    print(f"   • Épocas: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Validation split: {VALIDATION_SPLIT}")
    print()
    
    # Configurar callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        [user_ids, beer_ids, style_ids, brewery_ids, abv_values, ibu_values],
        ratings,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n   ✓ Entrenamiento completado")
    
    return history

def guardar_artefactos(model, mappings, history):
    """
    Guarda modelo y mappings
    
    Args:
        model: Modelo entrenado
        mappings: Diccionario con todos los mappings
        history: Historial de entrenamiento
    """
    print("\n💾 Guardando artefactos...")
    
    # Guardar modelo
    model.save(MODEL_PATH)
    print(f"   ✓ Modelo guardado: {MODEL_PATH}")
    
    # Agregar metadatos de entrenamiento
    mappings['embedding_dim'] = EMBEDDING_DIM
    mappings['trained_at'] = datetime.now().isoformat()
    
    # Guardar mappings
    with open(MAPPINGS_PATH, 'wb') as f:
        pickle.dump(mappings, f)
    print(f"   ✓ Mappings guardados: {MAPPINGS_PATH}")
    
    # Mostrar métricas finales
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"\n📊 Métricas finales:")
    print(f"   • Loss: {final_loss:.4f}")
    print(f"   • Val Loss: {final_val_loss:.4f}")
    print(f"   • MAE: {final_mae:.4f}")
    print(f"   • Val MAE: {final_val_mae:.4f}")

def train_model():
    """Función principal de entrenamiento"""
    print("=" * 60)
    print("ENTRENAMIENTO MODELO TWO-TOWER")
    print("=" * 60)
    print()
    
    try:
        # 1. Preparar datos
        (user_ids, beer_ids, style_ids, brewery_ids, abv_values, ibu_values, 
         ratings, mappings) = preparar_datos()
        
        # 2. Crear modelo
        model = crear_modelo(
            mappings['n_users'], 
            mappings['n_beers'], 
            mappings['n_styles'], 
            mappings['n_breweries'], 
            EMBEDDING_DIM
        )
        
        # 3. Entrenar
        history = entrenar_modelo(
            model, user_ids, beer_ids, style_ids, brewery_ids, 
            abv_values, ibu_values, ratings, EPOCHS, BATCH_SIZE
        )
        
        # 4. Guardar
        guardar_artefactos(model, mappings, history)
        
        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO EXITOSO")
        print("=" * 60)
        print("\n📦 Archivos generados:")
        print(f"   • {MODEL_PATH}")
        print(f"   • {MAPPINGS_PATH}")
        print()
        
        # Registrar completitud si fue disparado por retrain_manager
        if '--history-id' in sys.argv:
            history_id = sys.argv[sys.argv.index('--history-id') + 1]
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE retrain_history 
                SET status = 'completed', completed_at = datetime('now')
                WHERE id = ?
            """, [history_id])
            conn.commit()
            conn.close()
            
            # Resetear contador y apagar fine-tuning
            reset_contador_evaluaciones()
            
            # Apagar modelos fine-tuned
            import shutil
            if os.path.exists(USER_MODELS_DIR):
                shutil.rmtree(USER_MODELS_DIR)
                os.makedirs(USER_MODELS_DIR)
            
            print(f"✅ Retreinamento completado (ID: {history_id})")
            print("✅ Contador de evaluaciones reseteado")
            print("✅ Modelos fine-tuned eliminados")
        
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

# =============================================================================
# SECCIÓN: FINE-TUNING DE USUARIOS
# =============================================================================

def fine_tune_user_model(user_id, epochs=5, batch_size=32):
    """Fine-tuning del modelo para usuario específico"""
    
    # Cargar modelo global
    model = keras.models.load_model(MODEL_PATH)
    
    # Cargar mappings
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)
    
    user_to_idx = mappings['user_to_idx']
    beer_to_idx = mappings['beer_to_idx']
    style_to_idx = mappings['style_to_idx']
    brewery_to_idx = mappings['brewery_to_idx']
    
    # Cargar datos del usuario
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT r.beer_id, r.rating, c.style, c.brewery_id, c.abv, c.ibu
        FROM interaccion r
        JOIN cervezas c ON r.beer_id = c.beer_id
        WHERE r.user_id = ? AND r.rating > 0
    """, [user_id])
    
    data = cursor.fetchall()
    conn.close()
    
    if len(data) < 10:
        print(f"Usuario {user_id} no tiene suficientes datos para fine-tuning")
        return
    
    # Preparar datos
    user_idx = user_to_idx[user_id]
    user_ids = np.array([user_idx] * len(data), dtype=np.int32)
    beer_ids = np.array([beer_to_idx.get(row['beer_id'], 0) for row in data], dtype=np.int32)
    style_ids = np.array([style_to_idx.get(row['style'], 0) for row in data], dtype=np.int32)
    brewery_ids = np.array([brewery_to_idx.get(row['brewery_id'], 0) for row in data], dtype=np.int32)
    abv_values = np.array([row['abv'] if row['abv'] else 0.0 for row in data], dtype=np.float32)
    ibu_values = np.array([row['ibu'] if row['ibu'] else 0.0 for row in data], dtype=np.float32)
    ratings = np.array([row['rating'] / 5.0 for row in data], dtype=np.float32)
    
    # Normalizar
    abv_normalized = np.clip(abv_values / 20.0, 0, 1)
    ibu_normalized = np.clip(ibu_values / 100.0, 0, 1)
    
    # Fine-tuning
    model.fit(
        [user_ids, beer_ids, style_ids, brewery_ids, abv_normalized, ibu_normalized],
        ratings,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Guardar modelo personalizado
    os.makedirs(USER_MODELS_DIR, exist_ok=True)
    user_model_path = f"{USER_MODELS_DIR}user_{user_id}.keras"
    model.save(user_model_path)
    
    # Actualizar estado
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE model_updates 
        SET status = 'completed', last_update_date = datetime('now')
        WHERE user_id = ?
    """, [user_id])
    conn.commit()
    conn.close()
    
    print(f"Fine-tuning completado para {user_id}")

# =============================================================================
# SECCIÓN: GESTIÓN DE RETREINAMIENTO
# =============================================================================

def contar_evaluaciones_usuario(user_id):
    """Cuenta evaluaciones del usuario"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM interaccion WHERE user_id = ? AND rating > 0", [user_id])
    result = cursor.fetchone()
    conn.close()
    return result['total']

def usuario_en_modelo(user_id):
    """Verifica si usuario está en el modelo Two-Tower"""
    if not os.path.exists(MAPPINGS_PATH):
        return False
    
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)
    
    return user_id in mappings.get('user_to_idx', {})

def necesita_retrain_individual(user_id):
    """Verifica si usuario necesita retreinamento individual"""
    evaluaciones_actuales = contar_evaluaciones_usuario(user_id)
    
    if evaluaciones_actuales < MIN_EVALUACIONES_PARA_RETRAIN:
        return False
    
    if usuario_en_modelo(user_id):
        return False
    
    return True

def necesita_fine_tuning(user_id):
    """Verifica si usuario necesita fine-tuning"""
    if not usuario_en_modelo(user_id):
        return False
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Buscar última atualização
    cursor.execute("""
        SELECT evaluaciones_at_last_update 
        FROM model_updates 
        WHERE user_id = ?
    """, [user_id])
    
    result = cursor.fetchone()
    evaluaciones_en_ultima = result['evaluaciones_at_last_update'] if result else 0
    evaluaciones_actuales = contar_evaluaciones_usuario(user_id)
    
    conn.close()
    
    # Fine-tuning a cada 10 evaluaciones nuevas
    return (evaluaciones_actuales - evaluaciones_en_ultima) >= EVALUACIONES_PARA_FINE_TUNING

def necesita_retrain_global():
    """Verifica si necesita retreinamento global"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Verificar contador de evaluaciones
    cursor.execute("SELECT value FROM retrain_config WHERE key = 'evaluaciones_desde_ultimo_retrain'")
    result = cursor.fetchone()
    evaluaciones_nuevas = int(result['value']) if result else 0
    
    # Verificar fecha del último retreinamento
    cursor.execute("SELECT value FROM retrain_config WHERE key = 'ultimo_retrain_global'")
    result = cursor.fetchone()
    ultimo_retrain = datetime.fromisoformat(result['value']) if result else datetime.now()
    
    conn.close()
    
    # Trigger: 50 evaluaciones O 1 día
    return evaluaciones_nuevas >= EVALUACIONES_PARA_RETRAIN_GLOBAL or (datetime.now() - ultimo_retrain) >= timedelta(days=DIAS_PARA_RETRAIN_GLOBAL)

def trigger_retrain_global(reason="manual"):
    """Dispara retreinamento global en background"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Registrar inicio
    cursor.execute("""
        INSERT INTO retrain_history (retrain_type, trigger_reason, started_at, status)
        VALUES ('global', ?, datetime('now'), 'running')
    """, [reason])
    history_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Ejecutar en background
    subprocess.Popen([
        'python3', '-c', 'from models import train_model; train_model()',
        '--history-id', str(history_id)
    ])
    
    print(f"Retreinamento global iniciado (ID: {history_id})")
    return history_id

def trigger_fine_tuning(user_id):
    """Dispara fine-tuning para usuario específico"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    evaluaciones_actuales = contar_evaluaciones_usuario(user_id)
    
    # Registrar inicio
    cursor.execute("""
        INSERT OR REPLACE INTO model_updates (user_id, evaluaciones_at_last_update, last_update_date, status)
        VALUES (?, ?, datetime('now'), 'running')
    """, [user_id, evaluaciones_actuales])
    
    conn.commit()
    conn.close()
    
    # Ejecutar en background
    subprocess.Popen([
        'python3', '-c', f'from models import fine_tune_user_model; fine_tune_user_model("{user_id}")'
    ])
    
    print(f"Fine-tuning iniciado para {user_id}")

def incrementar_contador_evaluaciones():
    """Incrementa contador de evaluaciones nuevas"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE retrain_config 
        SET value = CAST((CAST(value AS INTEGER) + 1) AS TEXT),
            updated_at = datetime('now')
        WHERE key = 'evaluaciones_desde_ultimo_retrain'
    """)
    
    conn.commit()
    conn.close()

def reset_contador_evaluaciones():
    """Resetea contador após retreinamento global"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE retrain_config 
        SET value = '0',
            updated_at = datetime('now')
        WHERE key = 'evaluaciones_desde_ultimo_retrain'
    """)
    
    cursor.execute("""
        UPDATE retrain_config 
        SET value = datetime('now'),
            updated_at = datetime('now')
        WHERE key = 'ultimo_retrain_global'
    """)
    
    conn.commit()
    conn.close()

def verificar_y_disparar_retreinamiento(user_id):
    """
    Verifica condiciones y dispara retreinamiento si necesario
    Llamar después de cada evaluación
    """
    # 1. Incrementar contador global
    incrementar_contador_evaluaciones()
    
    # 2. Verificar si usuario necesita retreinamento individual
    if necesita_retrain_individual(user_id):
        trigger_retrain_global(f"usuario_nuevo_{user_id}")
        return
    
    # 3. Verificar si usuario necesita fine-tuning
    if necesita_fine_tuning(user_id):
        trigger_fine_tuning(user_id)
    
    # 4. Verificar si necesita retreinamento global
    if necesita_retrain_global():
        trigger_retrain_global("automatico_50_o_diario")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--user-id':
        user_id = sys.argv[2]
        fine_tune_user_model(user_id)
    else:
        train_model()
