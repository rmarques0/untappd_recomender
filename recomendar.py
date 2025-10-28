## version: 3.0 -- recomendaciones de cervejas usando SQLite

import sqlite3
import os
import random
from datetime import datetime
import pickle
import numpy as np

from config import DATABASE_FILE, MODEL_PATH, MAPPINGS_PATH, USER_MODELS_DIR
from database import get_db_connection
import utils as metricas

###

def sql_execute(query, params=None):
    """Ejecuta una consulta SQL que modifica datos"""
    con = sqlite3.connect(DATABASE_FILE)
    cur = con.cursor()
    if params:
        res = cur.execute(query, params)
    else:
        res = cur.execute(query)

    con.commit()
    con.close()
    return res

def sql_select(query, params=None):
    """Ejecuta una consulta SQL de selección"""
    con = sqlite3.connect(DATABASE_FILE)
    con.row_factory = sqlite3.Row # esto es para que devuelva registros en el fetchall
    cur = con.cursor()
    if params:
        res = cur.execute(query, params)
    else:
        res = cur.execute(query)

    ret = res.fetchall()
    con.close()
    return ret

###

def crear_usuario(user_id):
    """Crea un usuario en el sistema"""
    query = "INSERT INTO usuarios(user_id) VALUES (?) ON CONFLICT DO NOTHING;"
    sql_execute(query, [user_id])
    return

def insertar_interacciones(beer_id, user_id, rating):
    """Inserta o actualiza una interacción usuario-cerveza"""
    query = "INSERT INTO interaccion(beer_id, user_id, rating, fecha) VALUES (?, ?, ?, ?) ON CONFLICT (user_id, beer_id) DO UPDATE SET rating=?, fecha=?;"
    fecha = datetime.now().isoformat()
    sql_execute(query, [beer_id, user_id, rating, fecha, rating, fecha])
    return

def reset_usuario(user_id):
    """Resetea todas las interacciones de un usuario"""
    query = "DELETE FROM interaccion WHERE user_id = ?;"
    sql_execute(query, [user_id])
    return

def obtener_cerveza(beer_id):
    """Obtiene los datos de una cerveza específica"""
    query = "SELECT * FROM cervezas WHERE beer_id = ?;"
    result = sql_select(query, [beer_id])
    if result:
        cerveza = dict(result[0])
        # Asegurar que image_url existe
        if cerveza.get('image_url') is None:
            cerveza['image_url'] = ''
        return cerveza
    return None

def items_valorados(user_id):
    """Obtiene las cervezas que el usuario ha valorado (rating > 0)"""
    query = "SELECT beer_id FROM interaccion WHERE user_id = ? AND rating > 0"
    rows = sql_select(query, [user_id])
    return [row["beer_id"] for row in rows]

def items_vistos(user_id):
    """Obtiene las cervezas que el usuario ha visto (rating = 0)"""
    query = "SELECT beer_id FROM interaccion WHERE user_id = ? AND rating = 0"
    rows = sql_select(query, [user_id])
    return [row["beer_id"] for row in rows]

def items_desconocidos(user_id):
    """Obtiene las cervezas que el usuario no conoce"""
    query = """
    SELECT beer_id FROM cervezas 
    WHERE beer_id NOT IN (
        SELECT beer_id FROM interaccion 
        WHERE user_id = ? AND rating IS NOT NULL
    )
    """
    rows = sql_select(query, [user_id])
    return [row["beer_id"] for row in rows]

def datos_cervezas(beer_ids):
    """Obtiene los datos de múltiples cervezas"""
    if not beer_ids:
        return []
    
    placeholders = ','.join(['?'] * len(beer_ids))
    query = f"SELECT * FROM cervezas WHERE beer_id IN ({placeholders})"
    rows = sql_select(query, beer_ids)
    
    cervezas = []
    for row in rows:
        cerveza = dict(row)
        # Asegurar que image_url existe
        if cerveza.get('image_url') is None:
            cerveza['image_url'] = ''
        cervezas.append(cerveza)
    
    return cervezas

###

def recomendar_azar(user_id, cervezas_relevantes, cervezas_desconocidas, N=9):
    """Recomendación aleatoria (versión básica para cold start)"""
    if len(cervezas_desconocidas) < N:
        return cervezas_desconocidas
    return random.sample(cervezas_desconocidas, N)

def recomendar_popular(user_id, cervezas_desconocidas, N=9):
    """Recomendación basada en popularidad (para cold start)"""
    if not cervezas_desconocidas:
        return []
    
    # Obtener cervezas populares ordenadas por rating y cantidad de evaluaciones
    placeholders = ','.join(['?'] * len(cervezas_desconocidas))
    query = f"""
        SELECT beer_id, rating, total_ratings 
        FROM cervezas 
        WHERE beer_id IN ({placeholders})
        ORDER BY rating DESC, total_ratings DESC
        LIMIT ?
    """
    result = sql_select(query, cervezas_desconocidas + [N])
    return [row["beer_id"] for row in result]

def recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N=9):
    """Recomendación basada en filtrado colaborativo"""
    if len(cervezas_relevantes) < 3:  # Necesitamos al menos 3 evaluaciones
        return recomendar_popular(user_id, cervezas_desconocidas, N)
    
    # Obtener usuarios similares
    usuarios_similares = obtener_usuarios_similares(user_id, cervezas_relevantes)
    
    if not usuarios_similares:
        return recomendar_popular(user_id, cervezas_desconocidas, N)
    
    # Obtener cervezas recomendadas por usuarios similares
    cervezas_recomendadas = obtener_cervezas_usuarios_similares(usuarios_similares, cervezas_desconocidas, N)
    
    if len(cervezas_recomendadas) < N:
        # Complementar con cervezas populares si no hay suficientes
        cervezas_populares = recomendar_popular(user_id, cervezas_desconocidas, N - len(cervezas_recomendadas))
        cervezas_recomendadas.extend(cervezas_populares)
    
    return cervezas_recomendadas[:N]

def obtener_usuarios_similares(user_id, cervezas_relevantes, min_similarity=0.3, max_users=50):
    """Obtiene usuarios con gustos similares usando similitud de coseno"""
    if not cervezas_relevantes:
        return []
    
    # Obtener ratings del usuario actual
    user_ratings = {}
    for beer_id in cervezas_relevantes:
        query = "SELECT rating FROM interaccion WHERE user_id = ? AND beer_id = ?"
        result = sql_select(query, [user_id, beer_id])
        if result:
            user_ratings[beer_id] = result[0]["rating"]
    
    if not user_ratings:
        return []
    
    # Buscar usuarios que hayan evaluado al menos 2 de las mismas cervezas
    placeholders = ','.join(['?'] * len(cervezas_relevantes))
    query = f"""
        SELECT DISTINCT i1.user_id, i1.beer_id, i1.rating
        FROM interaccion i1
        WHERE i1.user_id != ? 
        AND i1.beer_id IN ({placeholders})
        AND i1.rating > 0
    """
    result = sql_select(query, [user_id] + cervezas_relevantes)
    
    # Agrupar por usuario
    user_ratings_dict = {}
    for row in result:
        other_user = row["user_id"]
        if other_user not in user_ratings_dict:
            user_ratings_dict[other_user] = {}
        user_ratings_dict[other_user][row["beer_id"]] = row["rating"]
    
    # Calcular similitud de coseno
    similar_users = []
    for other_user, other_ratings in user_ratings_dict.items():
        if len(other_ratings) < 2:  # Necesitamos al menos 2 cervezas en común
            continue
            
        similarity = calcular_similitud_coseno(user_ratings, other_ratings)
        if similarity >= min_similarity:
            similar_users.append((other_user, similarity))
    
    # Ordenar por similitud y tomar los mejores
    similar_users.sort(key=lambda x: x[1], reverse=True)
    return [user_id for user_id, _ in similar_users[:max_users]]

def calcular_similitud_coseno(ratings1, ratings2):
    """Calcula la similitud de coseno entre dos conjuntos de ratings"""
    # Encontrar cervezas comunes
    common_beers = set(ratings1.keys()) & set(ratings2.keys())
    
    if len(common_beers) < 2:
        return 0.0
    
    # Calcular productos punto y magnitudes
    dot_product = sum(ratings1[beer] * ratings2[beer] for beer in common_beers)
    magnitude1 = sum(rating ** 2 for rating in ratings1.values()) ** 0.5
    magnitude2 = sum(rating ** 2 for rating in ratings2.values()) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def obtener_cervezas_usuarios_similares(usuarios_similares, cervezas_desconocidas, N=9):
    """Obtiene cervezas recomendadas por usuarios similares"""
    if not usuarios_similares or not cervezas_desconocidas:
        return []
    
    # Obtener cervezas bien evaluadas por usuarios similares
    placeholders_users = ','.join(['?'] * len(usuarios_similares))
    placeholders_beers = ','.join(['?'] * len(cervezas_desconocidas))
    
    query = f"""
        SELECT i.beer_id, AVG(i.rating) as avg_rating, COUNT(*) as count_ratings
        FROM interaccion i
        WHERE i.user_id IN ({placeholders_users})
        AND i.beer_id IN ({placeholders_beers})
        AND i.rating > 0
        GROUP BY i.beer_id
        HAVING count_ratings >= 2
        ORDER BY avg_rating DESC, count_ratings DESC
        LIMIT ?
    """
    
    result = sql_select(query, usuarios_similares + cervezas_desconocidas + [N])
    return [row["beer_id"] for row in result]

def recomendar(user_id, cervezas_relevantes=None, cervezas_desconocidas=None, N=9):
    """Función principal de recomendación con transición automática"""
    if not cervezas_relevantes:
        cervezas_relevantes = items_valorados(user_id)

    if not cervezas_desconocidas:
        cervezas_desconocidas = items_desconocidos(user_id)

    # Estrategia de recomendación basada en cantidad de datos
    num_evaluaciones = len(cervezas_relevantes)
    
    if num_evaluaciones == 0:
        # Cold start: recomendaciones populares
        return recomendar_popular(user_id, cervezas_desconocidas, N)
    elif num_evaluaciones < 10:
        # Few-shot: mezcla de popular y colaborativo
        popular = recomendar_popular(user_id, cervezas_desconocidas, N)
        colaborativo = recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N)
        
        # Combinar y eliminar duplicados manteniendo el orden
        resultado = popular + colaborativo
        resultado_sin_duplicados = []
        for cerveza in resultado:
            if cerveza not in resultado_sin_duplicados:
                resultado_sin_duplicados.append(cerveza)
        
        # Si no tenemos suficientes, completar con más populares
        if len(resultado_sin_duplicados) < N:
            cervezas_restantes = [c for c in cervezas_desconocidas if c not in resultado_sin_duplicados]
            adicionales = recomendar_popular(user_id, cervezas_restantes, N - len(resultado_sin_duplicados))
            resultado_sin_duplicados.extend(adicionales)
        
        return resultado_sin_duplicados[:N]
    else:
        # Suficientes datos: usar two-tower si está disponible
        try:
            return recomendar_two_tower(user_id, N)
        except Exception as e:
            # Fallback a colaborativo si two-tower falla
            print(f"Two-tower falló ({e}), usando colaborativo")
            return recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N)

def recomendar_contexto(user_id, beer_id, cervezas_relevantes=None, cervezas_desconocidas=None, N=3):
    """Recomendación contextual basada en una cerveza específica"""
    if not cervezas_relevantes:
        cervezas_relevantes = items_valorados(user_id)

    if not cervezas_desconocidas:
        cervezas_desconocidas = items_desconocidos(user_id)

    # Usar la misma lógica de transición que la recomendación general
    return recomendar(user_id, cervezas_relevantes, cervezas_desconocidas, N)

###

def recomendar_two_tower(user_id, N=9):
    """
    Recomendación usando modelo Two-Tower con embeddings
    
    Args:
        user_id: ID del usuario
        N: Cantidad de recomendaciones a retornar
        
    Returns:
        list: Lista de beer_ids recomendados
        
    Raises:
        Exception: Si el modelo no existe o el usuario no está en el modelo
    """
    import pickle
    import numpy as np
    
    # Verificar si existe modelo fine-tuned para el usuario
    user_model_path = f"{USER_MODELS_DIR}/user_{user_id}.keras"
    model_path = user_model_path if os.path.exists(user_model_path) else MODEL_PATH
    
    # Verificar que modelo existe
    if not os.path.exists(model_path):
        raise Exception("Modelo two-tower no encontrado. Ejecutar train_two_tower.py primero.")
    
    if not os.path.exists(MAPPINGS_PATH):
        raise Exception("Mappings no encontrados. Ejecutar train_two_tower.py primero.")
    
    # Cargar modelo (global o fine-tuned)
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except ImportError:
        raise Exception("Keras no instalado. Agregar a requirements.txt")
    
    # Cargar mappings
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)
    
    user_to_idx = mappings['user_to_idx']
    beer_to_idx = mappings['beer_to_idx']
    idx_to_beer = mappings['idx_to_beer']
    style_to_idx = mappings['style_to_idx']
    brewery_to_idx = mappings['brewery_to_idx']
    
    # Verificar que usuario existe en modelo
    if user_id not in user_to_idx:
        print(f"⚠️  Usuario {user_id} no está en el modelo entrenado, usando estrategia de fallback")
        # Obtener cervezas desconocidas y relevantes para fallback
        cervezas_desconocidas = items_desconocidos(user_id)
        cervezas_relevantes = items_valorados(user_id)
        return recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N)
    
    # Obtener cervezas desconocidas y relevantes
    cervezas_desconocidas = items_desconocidos(user_id)
    cervezas_relevantes = items_valorados(user_id)
    
    if not cervezas_desconocidas:
        return []
    
    # Filtrar solo cervezas que están en el modelo
    cervezas_validas = [b for b in cervezas_desconocidas if b in beer_to_idx]
    
    if not cervezas_validas:
        # Si no hay cervezas en el modelo, usar estrategia de fallback
        print("⚠️  No hay cervezas desconocidas en el modelo, usando estrategia de fallback")
        return recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N)
    
    # Preparar inputs para predicción
    user_idx = user_to_idx[user_id]
    user_ids_array = np.array([user_idx] * len(cervezas_validas), dtype=np.int32)
    beer_ids_array = np.array([beer_to_idx[b] for b in cervezas_validas], dtype=np.int32)
    
    # Obtener features adicionales de las cervezas
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(cervezas_validas))
    cursor.execute(f"""
        SELECT beer_id, style, brewery_id, abv, ibu 
        FROM cervezas 
        WHERE beer_id IN ({placeholders})
    """, cervezas_validas)
    
    beer_features = {row['beer_id']: row for row in cursor.fetchall()}
    conn.close()
    
    # Preparar arrays de features categóricas y numéricas
    style_ids_array = np.array([
        style_to_idx.get(beer_features[b]['style'], 0) for b in cervezas_validas
    ], dtype=np.int32)
    
    brewery_ids_array = np.array([
        brewery_to_idx.get(beer_features[b]['brewery_id'], 0) for b in cervezas_validas
    ], dtype=np.int32)
    
    abv_values = np.array([
        beer_features[b]['abv'] if beer_features[b]['abv'] else 0.0 for b in cervezas_validas
    ], dtype=np.float32) / 20.0  # Normalizar ABV
    
    ibu_values = np.array([
        beer_features[b]['ibu'] if beer_features[b]['ibu'] else 0.0 for b in cervezas_validas
    ], dtype=np.float32) / 100.0  # Normalizar IBU
    
    # Hacer predicciones con todas las features
    scores = model.predict([
        user_ids_array, beer_ids_array, style_ids_array, brewery_ids_array, 
        abv_values, ibu_values
    ], verbose=0).flatten()
    
    # Ordenar por score descendente y tomar top-N
    top_indices = np.argsort(scores)[::-1][:N]
    top_beer_ids = [cervezas_validas[i] for i in top_indices]
    
    return top_beer_ids

###

def test(user_id):
    """Función de test para evaluar recomendaciones"""
    cervezas_relevantes = items_valorados(user_id)
    cervezas_desconocidas = items_vistos(user_id) + items_desconocidos(user_id)

    if len(cervezas_relevantes) < 10:  # Necesitamos suficientes datos
        return 0.0

    random.shuffle(cervezas_relevantes)

    corte = int(len(cervezas_relevantes) * 0.8)
    cervezas_relevantes_training = cervezas_relevantes[:corte]
    cervezas_relevantes_testing = cervezas_relevantes[corte:] + cervezas_desconocidas

    recomendacion = recomendar(user_id, cervezas_relevantes_training, cervezas_relevantes_testing, 20)

    relevance_scores = []
    for beer_id in recomendacion:
        query = "SELECT rating FROM interaccion WHERE user_id = ? AND beer_id = ?;"
        result = sql_select(query, [user_id, beer_id])
        if result and len(result) > 0:
            rating = result[0]["rating"]
        else:
            rating = 0
        relevance_scores.append(rating)
    
    score = metricas.normalized_discounted_cumulative_gain(relevance_scores)
    return score

if __name__ == '__main__':
    # Test con usuarios que tienen suficientes interacciones
    query = """
    SELECT user_id FROM usuarios 
    WHERE (SELECT COUNT(*) FROM interaccion WHERE user_id = usuarios.user_id) >= 10 
    LIMIT 10
    """
    users_with_data = sql_select(query)
    
    if not users_with_data:
        print("No hay usuarios con suficientes datos para test")
    else:
        scores = []
        for user_row in users_with_data:
            user_id = user_row["user_id"]
            score = test(user_id)
            scores.append(score)
            print(f"{user_id} >> {score:.6f}")

        if scores:
            print(f"NDCG promedio: {sum(scores)/len(scores):.6f}")