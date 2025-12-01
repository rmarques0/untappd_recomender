#!/usr/bin/env python3
"""
Evaluación de estrategias de recomendación
Calcula métricas de entrenamiento y test para cada estrategia
"""

import random
from database import get_db_connection
from utils import (
    normalized_discounted_cumulative_gain,
    precision_at_k,
    recall_at_k,
)

def evaluar_estrategia(user_id, estrategia, train_ratings, test_ratings, all_beers, train_data_global, k=10):
    """
    Evalúa una estrategia específica construyendo el mismo escenario que enfrenta en producción.
    
    Se generan candidatas tomando todas las cervezas que el usuario NO evaluó en el subset de
    entrenamiento y se fuerza la inclusión de los ítems de test para permitir medir hits reales.
    Los ratings de test se normalizan a la escala 0-1 (1-5 => 0-1) antes de calcular NDCG, lo
    que hace que todas las estrategias se comparen bajo el mismo criterio.
    
    Args:
        user_id: ID del usuario
        estrategia: 'zero-shot', 'hibrido', 'colaborativo', 'two-tower'
        train_ratings: Lista de (beer_id, rating) para entrenamiento
        test_ratings: Lista de (beer_id, rating) para test
        all_beers: Lista de todas las cervezas disponibles
        train_data_global: Dict con todos los ratings de treino {user_id: [(beer_id, rating), ...]}
        k: Top-K para métricas
        
    Returns:
        dict: Métricas calculadas (ndcg, precision, recall)
    """
    train_beer_ids = [beer_id for beer_id, _ in train_ratings]
    test_beer_ids = [beer_id for beer_id, _ in test_ratings]
    test_ratings_dict = {beer_id: rating for beer_id, rating in test_ratings}
    
    # METODOLOGÍA: Incluir TODAS las cervezas no evaluadas en el entrenamiento
    # Esto simula el escenario real donde el sistema recomienda entre todos los items desconocidos
    cervezas_candidatas = [
        b for b in all_beers 
        if b not in train_beer_ids  # Excluir solo las que el usuario ya evaluó en el entrenamiento
    ]
    
    # Garantizar que test_beer_ids estén siempre incluidos
    for beer_id in test_beer_ids:
        if beer_id not in cervezas_candidatas:
            cervezas_candidatas.append(beer_id)
    
    try:
        # Obtener recomendaciones según estrategia
        if estrategia == 'zero-shot':
            from recomendar import recomendar_popular
            try:
                recomendaciones = recomendar_popular(user_id, cervezas_candidatas, k)
                if not recomendaciones:
                    print(f"Advertencia: zero-shot retornó lista vacía para usuario {user_id}")
            except Exception as e:
                print(f"Error en zero-shot para usuario {user_id}: {e}")
                recomendaciones = []
        
        elif estrategia == 'hibrido':
            from recomendar import recomendar_popular
            recomendaciones = []
            
            try:
                colaborativo = recomendar_colaborativo_memoria(user_id, train_beer_ids, cervezas_candidatas, train_data_global, k)
                if colaborativo:
                    recomendaciones.extend(colaborativo[:k//2])
            except Exception as e:
                print(f"Error en colaborativo para híbrido (usuario {user_id}): {e}")
            
            try:
                popular = recomendar_popular(user_id, cervezas_candidatas, k)
                if popular:
                    cervezas_popular_sin_duplicados = [c for c in popular if c not in recomendaciones]
                    recomendaciones.extend(cervezas_popular_sin_duplicados[:k//2])
            except Exception as e:
                print(f"Error en popular para híbrido (usuario {user_id}): {e}")
            
            recomendaciones = list(dict.fromkeys(recomendaciones))[:k]
            
            if len(recomendaciones) < k:
                cervezas_restantes = [c for c in cervezas_candidatas if c not in recomendaciones]
                if cervezas_restantes:
                    try:
                        adicionales = recomendar_popular(user_id, cervezas_restantes, k - len(recomendaciones))
                        if adicionales:
                            recomendaciones.extend(adicionales)
                            recomendaciones = list(dict.fromkeys(recomendaciones))[:k]
                    except Exception as e:
                        print(f"Error completando híbrido (usuario {user_id}): {e}")
            
            if not recomendaciones:
                try:
                    recomendaciones = recomendar_popular(user_id, cervezas_candidatas, k)
                except Exception as e:
                    print(f"Error crítico en híbrido fallback (usuario {user_id}): {e}")
                    recomendaciones = []
        
        elif estrategia == 'colaborativo':
            try:
                recomendaciones = recomendar_colaborativo_memoria(user_id, train_beer_ids, cervezas_candidatas, train_data_global, k)
                if not recomendaciones:
                    from recomendar import recomendar_popular
                    print(f"Advertencia: colaborativo retornó lista vacía para usuario {user_id}, usando popular como fallback")
                    recomendaciones = recomendar_popular(user_id, cervezas_candidatas, k)
            except Exception as e:
                print(f"Error en colaborativo para usuario {user_id}: {e}")
                from recomendar import recomendar_popular
                try:
                    recomendaciones = recomendar_popular(user_id, cervezas_candidatas, k)
                except:
                    recomendaciones = []
        
        elif estrategia == 'two-tower':
            try:
                # Two-Tower necesita que las cervezas estén en el modelo
                # Filtrar solo cervezas que están en el modelo
                import pickle
                import os
                from config import MAPPINGS_PATH
                
                if os.path.exists(MAPPINGS_PATH):
                    with open(MAPPINGS_PATH, 'rb') as f:
                        mappings = pickle.load(f)
                    beer_to_idx = mappings.get('beer_to_idx', {})
                    user_to_idx = mappings.get('user_to_idx', {})
                    cervezas_validas = [b for b in cervezas_candidatas if b in beer_to_idx]
                    
                    if user_id not in user_to_idx:
                        raise ValueError(f"Usuario {user_id} no está en el modelo")
                    
                    if cervezas_validas:
                        recomendaciones = recomendar_two_tower_filtrado(user_id, cervezas_validas, k)
                        if not recomendaciones:
                            raise ValueError("Two-Tower retornou lista vazia")
                    else:
                        raise ValueError(f"No hay cervezas válidas ({len(cervezas_candidatas)} candidatas, {len(beer_to_idx)} en modelo)")
                else:
                    raise ValueError("MAPPINGS_PATH no existe")
            except Exception as e:
                print(f"Two-Tower fallback para usuario {user_id}: {e}")
                try:
                    recomendaciones = recomendar_colaborativo_memoria(user_id, train_beer_ids, cervezas_candidatas, train_data_global, k)
                    if not recomendaciones:
                        from recomendar import recomendar_popular
                        recomendaciones = recomendar_popular(user_id, cervezas_candidatas, k)
                except Exception as e2:
                    print(f"Error en fallback colaborativo para usuario {user_id}: {e2}")
                    from recomendar import recomendar_popular
                    try:
                        recomendaciones = recomendar_popular(user_id, cervezas_candidatas, k)
                    except:
                        recomendaciones = []
        
        else:
            recomendaciones = []
        
        # Calcular métricas
        relevance_scores = []
        relevant_items = set(test_beer_ids)
        
        for beer_id in recomendaciones:
            if beer_id in test_ratings_dict:
                # Normalizar rating 1-5 a 0-1 para NDCG
                rating = test_ratings_dict[beer_id]
                relevance_scores.append((rating - 1) / 4.0)
            else:
                relevance_scores.append(0.0)
        
        if not recomendaciones:
            return {'ndcg': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        ndcg = normalized_discounted_cumulative_gain(relevance_scores)
        precision = precision_at_k(recomendaciones, relevant_items, k)
        recall = recall_at_k(recomendaciones, relevant_items, k)
        
        return {'ndcg': ndcg, 'precision': precision, 'recall': recall}
    
    except Exception as e:
        print(f"Error evaluando estrategia {estrategia} para usuario {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return {'ndcg': 0.0, 'precision': 0.0, 'recall': 0.0}

def recomendar_two_tower_filtrado(user_id, cervezas_candidatas, N=9):
    """
    Versión de Two-Tower que filtra cervezas candidatas
    """
    import pickle
    import numpy as np
    import os
    from config import MODEL_PATH, MAPPINGS_PATH
    from database import get_db_connection
    
    # Cargar modelo
    import keras
    if not os.path.exists(MODEL_PATH):
        raise ValueError("Modelo two-tower no encontrado")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    
    # Cargar mappings
    if not os.path.exists(MAPPINGS_PATH):
        raise ValueError("Mappings no encontrados")
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)
    
    user_to_idx = mappings.get('user_to_idx', {})
    beer_to_idx = mappings.get('beer_to_idx', {})
    
    if user_id not in user_to_idx:
        raise ValueError(f"Usuario {user_id} no está en el modelo two-tower")
    
    # Filtrar solo cervezas candidatas que están en el modelo
    cervezas_validas = [b for b in cervezas_candidatas if b in beer_to_idx]
    
    if not cervezas_validas:
        raise ValueError("No hay cervezas candidatas presentes en el modelo two-tower")
    
    # Preparar inputs
    user_idx = user_to_idx[user_id]
    user_ids_array = np.array([user_idx] * len(cervezas_validas), dtype=np.int32)
    beer_ids_array = np.array([beer_to_idx[b] for b in cervezas_validas], dtype=np.int32)
    
    # Obtener features
    conn = get_db_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(cervezas_validas))
    cursor.execute(f"""
        SELECT beer_id, style, brewery_id, abv, ibu 
        FROM cervezas 
        WHERE beer_id IN ({placeholders})
    """, cervezas_validas)
    beer_features = {row['beer_id']: dict(row) for row in cursor.fetchall()}
    conn.close()
    
    # Filtrar solo cervezas que tienen features
    cervezas_com_features = [b for b in cervezas_validas if b in beer_features]
    if not cervezas_com_features:
        raise ValueError("Ninguna cerveza válida tiene features en el banco")
    
    if len(cervezas_com_features) < len(cervezas_validas):
        # Ajustar arrays para apenas cervezas com features
        cervezas_validas = cervezas_com_features
        user_ids_array = np.array([user_idx] * len(cervezas_validas), dtype=np.int32)
        beer_ids_array = np.array([beer_to_idx[b] for b in cervezas_validas], dtype=np.int32)
    
    style_to_idx = mappings.get('style_to_idx', {})
    brewery_to_idx = mappings.get('brewery_to_idx', {})
    
    style_ids_array = np.array([
        style_to_idx.get(beer_features[b].get('style'), 0) for b in cervezas_validas
    ], dtype=np.int32)
    
    brewery_ids_array = np.array([
        brewery_to_idx.get(beer_features[b].get('brewery_id'), 0) for b in cervezas_validas
    ], dtype=np.int32)
    
    abv_values = np.array([
        beer_features[b].get('abv') if beer_features[b].get('abv') else 0.0 for b in cervezas_validas
    ], dtype=np.float32) / 20.0
    
    ibu_values = np.array([
        beer_features[b].get('ibu') if beer_features[b].get('ibu') else 0.0 for b in cervezas_validas
    ], dtype=np.float32) / 100.0
    
    # Predicciones
    scores = model.predict([
        user_ids_array, beer_ids_array, style_ids_array, brewery_ids_array, 
        abv_values, ibu_values
    ], verbose=0).flatten()
    
    # Personalización: boost basado en historial del usuario
    # Obtener estilos y cervecerías que el usuario ya le gustaron (rating >= 4)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT c.style, c.brewery_id
        FROM ratings_historicos r
        JOIN cervezas c ON r.beer_id = c.beer_id
        WHERE r.username = ? AND r.rating >= 4
    """, [user_id])
    user_preferences = cursor.fetchall()
    conn.close()
    
    preferred_styles = set(row['style'] for row in user_preferences if row['style'])
    preferred_breweries = set(row['brewery_id'] for row in user_preferences if row['brewery_id'])
    
    # Aplicar boost para cervejas com estilos/cervejarias preferidos
    boosted_scores = scores.copy()
    for i, beer_id in enumerate(cervezas_validas):
        beer_info = beer_features.get(beer_id, {})
        style = beer_info.get('style')
        brewery_id = beer_info.get('brewery_id')
        
        boost = 0.0
        if style and style in preferred_styles:
            boost += 0.15  # Boost por estilo preferido
        if brewery_id and brewery_id in preferred_breweries:
            boost += 0.15  # Boost por cervejaria preferida
        
        boosted_scores[i] += boost
    
    # Top-N com scores boostados
    if len(boosted_scores) == 0:
        raise ValueError("Modelo retornou scores vazios")
    
    top_indices = np.argsort(boosted_scores)[::-1][:N]
    resultado = [cervezas_validas[i] for i in top_indices]
    
    if not resultado:
        raise ValueError("Nenhuma recomendação gerada")
    
    return resultado

def recomendar_colaborativo_memoria(user_id, cervezas_relevantes, cervezas_candidatas, train_data_global, N=9):
    """
    Colaborativo usando datos en memoria
    """
    if N <= 0 or not cervezas_candidatas:
        return []
    
    if len(cervezas_relevantes) < 3 or not train_data_global:
        from recomendar import recomendar_popular
        return recomendar_popular(user_id, cervezas_candidatas, N)
    
    try:
        # Obtener ratings del usuario
        user_ratings_dict = {beer_id: rating for beer_id, rating in train_data_global.get(user_id, []) if beer_id in cervezas_relevantes}
        
        if not user_ratings_dict:
            from recomendar import recomendar_popular
            return recomendar_popular(user_id, cervezas_candidatas, N)
        
        # Buscar usuarios similares (similitud de coseno)
        usuarios_similares = []
        for other_user, other_ratings in train_data_global.items():
            if other_user == user_id:
                continue
            other_ratings_dict = {beer_id: rating for beer_id, rating in other_ratings}
            common_beers = set(user_ratings_dict.keys()) & set(other_ratings_dict.keys())
            
            if len(common_beers) >= 2:
                dot_product = sum(user_ratings_dict[b] * other_ratings_dict[b] for b in common_beers)
                mag1 = sum(r**2 for r in user_ratings_dict.values()) ** 0.5
                mag2 = sum(r**2 for r in other_ratings_dict.values()) ** 0.5
                if mag1 > 0 and mag2 > 0:
                    similarity = dot_product / (mag1 * mag2)
                    if similarity >= 0.3:
                        usuarios_similares.append((other_user, similarity))
        
        usuarios_similares.sort(key=lambda x: x[1], reverse=True)
        usuarios_similares = [uid for uid, _ in usuarios_similares[:50]]
        
        if not usuarios_similares:
            from recomendar import recomendar_popular
            return recomendar_popular(user_id, cervezas_candidatas, N)
        
        # Agregar cervezas de usuarios similares
        beer_scores = {}
        for similar_user in usuarios_similares:
            for beer_id, rating in train_data_global.get(similar_user, []):
                if beer_id in cervezas_candidatas and rating >= 3:
                    if beer_id not in beer_scores:
                        beer_scores[beer_id] = []
                    beer_scores[beer_id].append(rating)
        
        # Calcular scores
        import math
        scored_beers = []
        for beer_id, ratings in beer_scores.items():
            avg_rating = sum(ratings) / len(ratings)
            count = len(ratings)
            score = avg_rating * (1 + math.log(count + 1))
            scored_beers.append((beer_id, score))
        
        scored_beers.sort(key=lambda x: x[1], reverse=True)
        recomendaciones = [beer_id for beer_id, _ in scored_beers[:N]]
        
        if len(recomendaciones) < N:
            from recomendar import recomendar_popular
            cervezas_restantes = [c for c in cervezas_candidatas if c not in recomendaciones]
            if cervezas_restantes:
                popular = recomendar_popular(user_id, cervezas_restantes, N - len(recomendaciones))
                if popular:
                    recomendaciones.extend(popular)
        
        if not recomendaciones:
            from recomendar import recomendar_popular
            return recomendar_popular(user_id, cervezas_candidatas, N)
        
        return recomendaciones[:N]
    except Exception as e:
        print(f"Error en recomendar_colaborativo_memoria para usuario {user_id}: {e}")
        from recomendar import recomendar_popular
        return recomendar_popular(user_id, cervezas_candidatas, N)

def evaluar_estrategias(n_usuarios=500, k=10, test_ratio=0.2):
    """
    Ejecuta evaluación cruzada user-based con split 80/20 por usuario.
    
    Cada usuario válido tiene sus ratings barajados y se separan en train/test usando el
    parámetro `test_ratio`. Durante la evaluación se replican los pasos productivos:
        1. Métricas en test con candidatas globales (all_beers - train) y top-K definido.
        2. Métricas en train usando un split interno 80/20 para monitorear overfitting.
    
    Args:
        n_usuarios: Número de usuarios a evaluar
        k: Top-K para métricas
        test_ratio: Proporción para test (default 0.2 = 20%)
        
    Returns:
        dict: Resultados por estrategia
    """
    print("=" * 60)
    print("EVALUACIÓN DE ESTRATEGIAS DE RECOMENDACIÓN")
    print("=" * 60)
    
    print("\n📊 Paso 1: Cargando datos históricos...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT username as user_id, beer_id, rating, date
        FROM ratings_historicos 
        WHERE rating > 0
    """)
    all_ratings = cursor.fetchall()
    
    # CORRECCIÓN: Usar solo cervezas que tienen ratings (no todas las 12420)
    # De las 12420 cervezas, solo 2254 tienen ratings (18%)
    # Usar todas las 12420 como candidatas es irrealista y diluye las métricas
    cursor.execute("SELECT DISTINCT beer_id FROM ratings_historicos WHERE rating > 0")
    all_beers = [row['beer_id'] for row in cursor.fetchall()]
    conn.close()
    
    print(f"   ✓ {len(all_beers)} cervezas com ratings (candidatas realistas)")
    
    # Agrupar por usuario (incluindo date para split temporal)
    ratings_por_usuario = {}
    for row in all_ratings:
        user_id = row['user_id']
        if user_id not in ratings_por_usuario:
            ratings_por_usuario[user_id] = []
        # Acessar date diretamente (pode ser None)
        date_val = row['date'] if 'date' in row.keys() else None
        ratings_por_usuario[user_id].append((row['beer_id'], row['rating'], date_val))
    
    # Filtrar usuarios con >=20 ratings para ter split mais robusto
    # Con >=10, el test tiene solo 2 cervezas en promedio, lo cual es muy poco
    # Com >=20, o test tem ~4 cervezas, ainda pouco mas mais realista
    min_ratings = 20  # Aumentar para ter mais dados no test
    usuarios_validos = {uid: ratings for uid, ratings in ratings_por_usuario.items() if len(ratings) >= min_ratings}
    
    # Split 80/20 por usuario - CORREGIDO: Split por cerveza única, no por rating
    # Esto evita que la misma cerveza aparezca en train y test
    train_data_global = {}
    test_data_global = {}
    
    for user_id, ratings in usuarios_validos.items():
        # Agrupar por cerveza única con timestamps para split temporal
        beer_ratings = {}
        for beer_id, rating, date in ratings:
            if beer_id not in beer_ratings:
                beer_ratings[beer_id] = []
            beer_ratings[beer_id].append((rating, date))
        
        # Para cada cerveza, usar rating medio y fecha más reciente
        unique_beer_ratings = []
        for beer_id, rating_dates in beer_ratings.items():
            avg_rating = sum(r for r, _ in rating_dates) / len(rating_dates)
            # Obtener fecha más reciente (o None si no hay)
            latest_date = max((d for _, d in rating_dates if d), default=None)
            unique_beer_ratings.append((beer_id, avg_rating, latest_date))
        
        # Split TEMPORAL: ordenar por data e usar os mais recentes como test
        # Si no hay fechas, usar split aleatorio como fallback
        if any(d for _, _, d in unique_beer_ratings if d):
            # Ordenar por data (mais antigas primeiro)
            unique_beer_ratings.sort(key=lambda x: x[2] if x[2] else '')
            split_idx = int(len(unique_beer_ratings) * (1 - test_ratio))
            train_data_global[user_id] = [(b, r) for b, r, _ in unique_beer_ratings[:split_idx]]
            test_data_global[user_id] = [(b, r) for b, r, _ in unique_beer_ratings[split_idx:]]
        else:
            # Fallback: split aleatorio si no hay fechas
            random.shuffle(unique_beer_ratings)
            split_idx = int(len(unique_beer_ratings) * (1 - test_ratio))
            train_data_global[user_id] = [(b, r) for b, r, _ in unique_beer_ratings[:split_idx]]
            test_data_global[user_id] = [(b, r) for b, r, _ in unique_beer_ratings[split_idx:]]
    
    print(f"   ✓ {len(usuarios_validos)} usuarios con >={min_ratings} ratings")
    print(f"   ✓ Split: {len(train_data_global)} usuarios en treino, {len(test_data_global)} en teste")
    
    # Estadísticas del split
    avg_train = sum(len(ratings) for ratings in train_data_global.values()) / len(train_data_global) if train_data_global else 0
    avg_test = sum(len(ratings) for ratings in test_data_global.values()) / len(test_data_global) if test_data_global else 0
    print(f"   ✓ Promedio: {avg_train:.1f} cervezas en train, {avg_test:.1f} en test por usuario")
    
    # Seleccionar usuarios de teste
    test_users = list(test_data_global.keys())[:n_usuarios]
    
    print(f"\n📈 Paso 2: Evaluando {len(test_users)} usuarios...")
    print(f"   Métricas: NDCG@{k}, Precision@{k}, Recall@{k}\n")
    
    estrategias = ['zero-shot', 'hibrido', 'colaborativo', 'two-tower']
    resultados = {estrategia: {
        'treino': {'ndcg': [], 'precision': [], 'recall': []},
        'teste': {'ndcg': [], 'precision': [], 'recall': []},
        'usuarios': 0
    } for estrategia in estrategias}
    
    for idx, user_id in enumerate(test_users, 1):
        train_ratings = train_data_global.get(user_id, [])
        test_ratings = test_data_global.get(user_id, [])
        
        if len(test_ratings) == 0:
            continue
        
        # Evaluar cada estrategia
        for estrategia in estrategias:
            try:
                # TESTE
                metricas_teste = evaluar_estrategia(
                    user_id, estrategia, train_ratings, test_ratings, 
                    all_beers, train_data_global, k
                )
                
                # TREINO (usar 80% do treino como treino, 20% como teste interno)
                train_split = int(len(train_ratings) * 0.8)
                train_train = train_ratings[:train_split]
                train_test = train_ratings[train_split:]
                
                metricas_treino = evaluar_estrategia(
                    user_id, estrategia, train_train, train_test,
                    all_beers, train_data_global, k
                )
                
                resultados[estrategia]['treino']['ndcg'].append(metricas_treino['ndcg'])
                resultados[estrategia]['treino']['precision'].append(metricas_treino['precision'])
                resultados[estrategia]['treino']['recall'].append(metricas_treino['recall'])
                
                resultados[estrategia]['teste']['ndcg'].append(metricas_teste['ndcg'])
                resultados[estrategia]['teste']['precision'].append(metricas_teste['precision'])
                resultados[estrategia]['teste']['recall'].append(metricas_teste['recall'])
                
                resultados[estrategia]['usuarios'] += 1
                
            except Exception as e:
                continue
        
        if idx % 100 == 0:
            print(f"   Procesados: {idx}/{len(test_users)} usuarios...")
    
    print(f"\n✅ Evaluación completa: {len(test_users)} usuarios")
    
    # Diagnóstico: Verificar cobertura del modelo Two-Tower
    print("\n" + "=" * 60)
    print("DIAGNÓSTICO TWO-TOWER")
    print("=" * 60)
    
    import pickle
    import os
    from config import MAPPINGS_PATH
    
    if os.path.exists(MAPPINGS_PATH):
        with open(MAPPINGS_PATH, 'rb') as f:
            mappings = pickle.load(f)
        beer_to_idx = mappings.get('beer_to_idx', {})
        user_to_idx = mappings.get('user_to_idx', {})
        
        total_test_beers = 0
        test_beers_in_model = 0
        usuarios_in_model = 0
        usuarios_not_in_model = 0
        
        for user_id in test_users[:100]:  # Muestra de 100 para diagnóstico
            test_ratings = test_data_global.get(user_id, [])
            test_beer_ids = [beer_id for beer_id, _ in test_ratings]
            
            total_test_beers += len(test_beer_ids)
            test_beers_in_model += len([b for b in test_beer_ids if b in beer_to_idx])
            
            if user_id in user_to_idx:
                usuarios_in_model += 1
            else:
                usuarios_not_in_model += 1
        
        print(f"Usuarios en modelo: {usuarios_in_model}/{usuarios_in_model + usuarios_not_in_model} (amostra)")
        print(f"Cervejas do teste no modelo: {test_beers_in_model}/{total_test_beers} ({test_beers_in_model/total_test_beers*100:.1f}%)")
        print(f"Total cervezas no modelo: {len(beer_to_idx)}")
    
    # Calcular promedios
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    
    resultados_finales = {}
    for estrategia in estrategias:
        if resultados[estrategia]['usuarios'] > 0:
            avg_ndcg_treino = sum(resultados[estrategia]['treino']['ndcg']) / len(resultados[estrategia]['treino']['ndcg'])
            avg_precision_treino = sum(resultados[estrategia]['treino']['precision']) / len(resultados[estrategia]['treino']['precision'])
            avg_recall_treino = sum(resultados[estrategia]['treino']['recall']) / len(resultados[estrategia]['treino']['recall'])
            
            avg_ndcg_teste = sum(resultados[estrategia]['teste']['ndcg']) / len(resultados[estrategia]['teste']['ndcg'])
            avg_precision_teste = sum(resultados[estrategia]['teste']['precision']) / len(resultados[estrategia]['teste']['precision'])
            avg_recall_teste = sum(resultados[estrategia]['teste']['recall']) / len(resultados[estrategia]['teste']['recall'])
            
            resultados_finales[estrategia] = {
                'treino': {'ndcg': avg_ndcg_treino, 'precision': avg_precision_treino, 'recall': avg_recall_treino},
                'teste': {'ndcg': avg_ndcg_teste, 'precision': avg_precision_teste, 'recall': avg_recall_teste},
                'usuarios': resultados[estrategia]['usuarios']
            }
            
            print(f"\n{estrategia.upper().replace('-', ' ').title()}:")
            print(f"  Usuarios: {resultados[estrategia]['usuarios']}")
            print(f"  TREINO - NDCG@{k}: {avg_ndcg_treino:.4f} | Precision@{k}: {avg_precision_treino:.4f} | Recall@{k}: {avg_recall_treino:.4f}")
            print(f"  TESTE  - NDCG@{k}: {avg_ndcg_teste:.4f} | Precision@{k}: {avg_precision_teste:.4f} | Recall@{k}: {avg_recall_teste:.4f}")
    
    return resultados_finales

if __name__ == '__main__':
    resultados = evaluar_estrategias(n_usuarios=500, k=10)
    
    print("\n" + "=" * 60)
    print("TABLA RESUMEN")
    print("=" * 60)
    print("\n| Estrategia | Treino NDCG | Treino Prec | Treino Rec | Teste NDCG | Teste Prec | Teste Rec |")
    print("|-----------|-------------|-------------|------------|------------|------------|-----------|")
    
    estrategias_nombres = {
        'zero-shot': 'Zero-shot',
        'hibrido': 'Híbrido',
        'colaborativo': 'Colaborativo',
        'two-tower': 'Two-Tower'
    }
    
    for estrategia, datos in resultados.items():
        nombre = estrategias_nombres.get(estrategia, estrategia)
        print(f"| {nombre:11} | {datos['treino']['ndcg']:11.4f} | {datos['treino']['precision']:11.4f} | {datos['treino']['recall']:10.4f} | {datos['teste']['ndcg']:10.4f} | {datos['teste']['precision']:10.4f} | {datos['teste']['recall']:9.4f} |")

