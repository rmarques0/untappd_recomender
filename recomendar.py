## version: 3.0 -- recomendaciones de cervejas usando SQLite

import sqlite3
import os
import random
from datetime import datetime
import pickle
import numpy as np

from config import DATABASE_FILE, MODEL_PATH, MAPPINGS_PATH, USER_MODELS_DIR
from database import get_db_connection
from utils import normalized_discounted_cumulative_gain

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
    rows = sql_select(query, tuple(beer_ids))
    
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
    if not cervezas_desconocidas or N <= 0:
        return []
    
    try:
        # Obtener top 30 para tener variedad y rotación
        top_k = max(30, N * 3)
    placeholders = ','.join(['?'] * len(cervezas_desconocidas))
    query = f"""
        SELECT beer_id, rating, total_ratings 
        FROM cervezas 
        WHERE beer_id IN ({placeholders})
            ORDER BY rating DESC, total_ratings DESC, beer_id ASC
            LIMIT {top_k}
    """
        result = sql_select(query, tuple(cervezas_desconocidas))
    cervezas_encontradas = [row["beer_id"] for row in result]
    
        if len(cervezas_encontradas) <= N:
            return cervezas_encontradas
        
        # Usar hash del user_id para selección determinística pero variada
        import hashlib
        user_hash = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        start_idx = user_hash % (len(cervezas_encontradas) - N + 1)
        
        return cervezas_encontradas[start_idx:start_idx + N]
    except Exception as e:
        print(f"Error en recomendar_popular para usuario {user_id}: {e}")
        return []

def recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N=9):
    """Recomendación basada en filtrado colaborativo"""
    if N <= 0 or not cervezas_desconocidas:
        return []
    
    if len(cervezas_relevantes) < 3:
        return recomendar_popular(user_id, cervezas_desconocidas, N)
    
    try:
    usuarios_similares = obtener_usuarios_similares(user_id, cervezas_relevantes)
    
    if not usuarios_similares:
        return recomendar_popular(user_id, cervezas_desconocidas, N)
    
    cervezas_recomendadas = obtener_cervezas_usuarios_similares(usuarios_similares, cervezas_desconocidas, N)
    
    if len(cervezas_recomendadas) < N:
            cervezas_restantes = [c for c in cervezas_desconocidas if c not in cervezas_recomendadas]
            if cervezas_restantes:
                cervezas_populares = recomendar_popular(user_id, cervezas_restantes, N - len(cervezas_recomendadas))
                if cervezas_populares:
        cervezas_recomendadas.extend(cervezas_populares)
    
        if not cervezas_recomendadas:
            return recomendar_popular(user_id, cervezas_desconocidas, N)
        
    return cervezas_recomendadas[:N]
    except Exception as e:
        print(f"Error en recomendar_colaborativo para usuario {user_id}: {e}")
        return recomendar_popular(user_id, cervezas_desconocidas, N)

def obtener_usuarios_similares(user_id, cervezas_relevantes, min_similarity=0.3, max_users=50):
    """Obtiene usuarios con gustos similares usando similitud de coseno"""
    if not cervezas_relevantes:
        return []
    
    # Obtener ratings del usuario actual
    user_ratings = {}
    for beer_id in cervezas_relevantes:
        query = "SELECT rating FROM ratings_historicos WHERE username = ? AND beer_id = ?"
        result = sql_select(query, [user_id, beer_id])
        if result:
            user_ratings[beer_id] = result[0]["rating"]
    
    if not user_ratings:
        return []
    
    # Buscar usuarios que hayan evaluado al menos 2 de las mismas cervezas
    placeholders = ','.join(['?'] * len(cervezas_relevantes))
    query = f"""
        SELECT DISTINCT r.user_id as user_id, r.beer_id, r.rating
        FROM ratings_historicos r
        WHERE r.user_id != ? 
        AND r.beer_id IN ({placeholders})
        AND r.rating > 0
    """
    result = sql_select(query, tuple([user_id] + cervezas_relevantes))
    
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
        SELECT r.beer_id, AVG(r.rating) as avg_rating, COUNT(*) as count_ratings
        FROM ratings_historicos r
        WHERE r.user_id IN ({placeholders_users})
        AND r.beer_id IN ({placeholders_beers})
        AND r.rating > 0
        GROUP BY r.beer_id
        HAVING count_ratings >= 2
        ORDER BY avg_rating DESC, count_ratings DESC
        LIMIT ?
    """
    
    result = sql_select(query, tuple(usuarios_similares + cervezas_desconocidas + [N]))
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
        resultado = recomendar_popular(user_id, cervezas_desconocidas, N)
        return resultado, "Zero-shot (cervezas más valoradas)"
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
        
        return resultado_sin_duplicados[:N], "Híbrido (popular + colaborativo)"
    else:
        # Suficientes datos: usar two-tower si está disponible
        try:
            resultado = recomendar_two_tower(user_id, N)
            return resultado, "Two-Tower (personalizado)"
        except Exception as e:
            # Fallback a colaborativo si two-tower falla
            print(f"Two-tower falló ({e}), usando colaborativo")
            
            # Si el error es porque el usuario no está en el modelo, disparar retreinamento
            if "no está en el modelo" in str(e):
                try:
                    from models import trigger_retrain_global
                    trigger_retrain_global("user_not_in_model_retry")
                    print("Retreinamento global iniciado para incluir usuario")
                except Exception as retrain_error:
                    print(f"Error disparando retreinamento: {retrain_error}")
            elif "Modelo incompatible" in str(e) or "Retreinar modelo necesario" in str(e):
                try:
                    from models import trigger_retrain_global
                    trigger_retrain_global("model_incompatible")
                    print("Retreinamento global iniciado por incompatibilidad de modelo")
                except Exception as retrain_error:
                    print(f"Error disparando retreinamento: {retrain_error}")
            
            resultado = recomendar_colaborativo(user_id, cervezas_relevantes, cervezas_desconocidas, N)
            return resultado, "Colaborativo (basado en usuarios similares)"

def recomendar_contexto(user_id, beer_id, cervezas_relevantes=None, cervezas_desconocidas=None, N=6):
    """Recomendación contextual basada en una cerveza específica"""
    # Obtener información de la cerveza actual
    cerveza_actual = obtener_cerveza(beer_id)
    if not cerveza_actual:
        # Si no existe, usar recomendación normal
        if not cervezas_relevantes:
            cervezas_relevantes = items_valorados(user_id)
        if not cervezas_desconocidas:
            cervezas_desconocidas = items_desconocidos(user_id)
        resultado, sistema = recomendar(user_id, cervezas_relevantes, cervezas_desconocidas, N)
        # Excluir la cerveza actual si está en los resultados
        resultado = [b for b in resultado if b != beer_id]
        return resultado[:N], sistema
    
    # Obtener cervezas desconocidas excluyendo la actual
    if not cervezas_desconocidas:
        todas_desconocidas = items_desconocidos(user_id)
        cervezas_desconocidas = [b for b in todas_desconocidas if b != beer_id]
    else:
        cervezas_desconocidas = [b for b in cervezas_desconocidas if b != beer_id]
    
    if not cervezas_relevantes:
        cervezas_relevantes = items_valorados(user_id)
    
    # Buscar más cervezas similares de las que necesitamos para poder hacer shuffle
    cervezas_similares = buscar_cervezas_similares(cerveza_actual, cervezas_desconocidas, N * 2)
    
    # Si encontramos suficientes similares, hacer shuffle y tomar N
    if len(cervezas_similares) >= N:
        random.shuffle(cervezas_similares)
        return cervezas_similares[:N], "Similar (mismo estilo/cervecería)"
    
    # Si no hay suficientes similares, complementar con recomendación normal
    resultado_normal, sistema = recomendar(user_id, cervezas_relevantes, cervezas_desconocidas, N * 2)
    resultado_normal = [b for b in resultado_normal if b != beer_id]
    
    # Combinar similares con recomendación normal, evitando duplicados
    resultado_final = cervezas_similares.copy()
    for cerveza in resultado_normal:
        if cerveza not in resultado_final:
            resultado_final.append(cerveza)
    
    # Si aún no tenemos suficientes, buscar más similares
    if len(resultado_final) < N:
        cervezas_restantes = [b for b in cervezas_desconocidas if b not in resultado_final]
        adicionales = buscar_cervezas_similares(cerveza_actual, cervezas_restantes, N - len(resultado_final))
        resultado_final.extend(adicionales)
    
    # Hacer shuffle del resultado final para variar las recomendaciones
    random.shuffle(resultado_final)
    
    return resultado_final[:N], f"Similar + {sistema}"

def buscar_cervezas_similares(cerveza_actual, cervezas_desconocidas, N=6):
    """Busca cervezas similares basadas en estilo y cervecería"""
    if not cervezas_desconocidas:
        return []
    
    style = cerveza_actual.get('style')
    brewery_name = cerveza_actual.get('brewery_name')
    beer_id_actual = cerveza_actual.get('beer_id')
    
    # Filtrar la cerveza actual
    cervezas_desconocidas = [b for b in cervezas_desconocidas if b != beer_id_actual]
    
    if not cervezas_desconocidas:
        return []
    
    placeholders = ','.join(['?'] * len(cervezas_desconocidas))
    
    # Buscar más cervezas de las necesarias para poder hacer shuffle después
    limit = min(N * 3, len(cervezas_desconocidas))  # Buscar hasta 3x más para tener opciones
    
    # Buscar cervezas similares priorizando: mismo estilo + misma cervecería > mismo estilo > misma cervecería
    query = f"""
        SELECT beer_id, 
               CASE 
                   WHEN style = ? AND brewery_name = ? THEN 3
                   WHEN style = ? THEN 2
                   WHEN brewery_name = ? THEN 1
                   ELSE 0
               END as similarity_score,
               rating, total_ratings
        FROM cervezas 
        WHERE beer_id IN ({placeholders})
        AND beer_id != ?
        ORDER BY similarity_score DESC, rating DESC, total_ratings DESC
        LIMIT ?
    """
    
    query_params = [
        style or '', brewery_name or '',  # Para CASE WHEN (score 3)
        style or '',  # Para CASE WHEN (score 2)
        brewery_name or '',  # Para CASE WHEN (score 1)
    ] + cervezas_desconocidas + [
        beer_id_actual,  # Excluir la cerveza actual
        limit
    ]
    
    result = sql_select(query, tuple(query_params))
    cervezas_encontradas = [row["beer_id"] for row in result]
    
    # Si tenemos más de N, hacer shuffle para variar
    if len(cervezas_encontradas) > N:
        random.shuffle(cervezas_encontradas)
    
    return cervezas_encontradas

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
    except Exception as e:
        if "Could not deserialize" in str(e) or "parent module" in str(e):
            raise Exception("Modelo incompatible con versión actual de Keras. Retreinar modelo necesario.")
        else:
            raise Exception(f"Error cargando modelo: {e}")
    
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
        print(f"Usuario {user_id} no está en el modelo entrenado, disparando retreinamento global")
        # Disparar retreinamento global para incluir este usuario
        try:
            from models import trigger_retrain_global
            trigger_retrain_global("new_user_not_in_model")
            print("Retreinamento global iniciado en background")
        except Exception as e:
            print(f"Error disparando retreinamento: {e}")
        
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
        print("Advertencia: No hay cervezas desconocidas en el modelo, usando estrategia de fallback")
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
    """
    Función de test para evaluar recomendaciones.
    
    Normaliza los ratings reales a la escala 0-1 (1-5 → 0-1) antes de calcular
    NDCG para mantener consistencia con evaluar.py y con las métricas del reporte.
    """
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
        normalized_rating = (rating - 1) / 4.0 if rating and rating > 0 else 0.0
        relevance_scores.append(normalized_rating)
    
    score = normalized_discounted_cumulative_gain(relevance_scores)
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