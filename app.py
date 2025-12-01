from flask import Flask, request, render_template, make_response, redirect, url_for, jsonify
import recomendar
from config import DATABASE_FILE, ADMIN_USER_ID, LAST_UPDATE_DATE
from database import get_db_connection
from models import verificar_y_disparar_retreinamiento, trigger_retrain_global

app = Flask(__name__)
app.debug = True

def is_first_visit(user_id):
    """Verifica si es la primera visita del usuario"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM interaccion WHERE user_id = ?", [user_id])
    count = cursor.fetchone()['count']
    conn.close()
    return count == 0

@app.get('/')
def get_index():
    return render_template('login.html')

@app.post('/')
def post_index():
    user_id = request.form.get('user_id', None)

    if user_id: # si me mandaron el user_id
        recomendar.crear_usuario(user_id)

        # mando al usuario a la página de recomendaciones
        res = make_response(redirect("/recomendaciones"))

        # pongo el user_id en una cookie para recordarlo
        res.set_cookie('user_id', user_id)
        return res

    # sino, le muestro el formulario de login
    return render_template('login.html')

@app.get('/recomendaciones')
def get_recomendaciones():
    user_id = request.cookies.get('user_id')
    
    # Verificar si es primera visita
    first_visit = is_first_visit(user_id)
    
    id_cervezas, sistema_usado = recomendar.recomendar(user_id)

    # pongo cervezas vistas con rating = 0
    for id_cerveza in id_cervezas:
        recomendar.insertar_interacciones(id_cerveza, user_id, 0)

    cervezas_recomendadas = recomendar.datos_cervezas(id_cervezas)
    cant_valoradas = len(recomendar.items_valorados(user_id))
    cant_vistas = len(recomendar.items_vistos(user_id))
    
    # Obtener ratings existentes del usuario para las cervezas recomendadas
    conn = get_db_connection()
    cursor = conn.cursor()
    user_ratings = {}
    if id_cervezas:
        placeholders = ','.join(['?'] * len(id_cervezas))
        cursor.execute(f"""
            SELECT beer_id, rating 
            FROM interaccion 
            WHERE user_id = ? AND beer_id IN ({placeholders})
        """, [user_id] + id_cervezas)
        for row in cursor.fetchall():
            rating_value = row['rating']
            # Asegurar que convertimos correctamente, incluso si es None o 0
            if rating_value is not None:
                user_ratings[row['beer_id']] = int(float(rating_value))
            else:
                user_ratings[row['beer_id']] = 0
    
    # Obtener estilos únicos para filtros
    cursor.execute("SELECT DISTINCT style FROM cervezas WHERE style IS NOT NULL AND style != '' ORDER BY style")
    estilos = [row['style'] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT brewery_name FROM cervezas WHERE brewery_name IS NOT NULL AND brewery_name != '' ORDER BY brewery_name")
    cervecerias = [row['brewery_name'] for row in cursor.fetchall()]
    conn.close()

    return render_template("recomendaciones.html", 
                         cervezas_recomendadas=cervezas_recomendadas, 
                         user_ratings=user_ratings,
                         user_id=user_id, 
                         cant_valoradas=cant_valoradas, 
                         cant_vistas=cant_vistas,
                         first_visit=first_visit,
                         estilos=estilos,
                         cervecerias=cervecerias,
                         sistema_usado=sistema_usado,
                         last_update_date=LAST_UPDATE_DATE)

@app.get('/cerveza/<string:id_cerveza>')
def get_cerveza_detalle(id_cerveza):
    """Página detallada de una cerveza específica"""
    user_id = request.cookies.get('user_id')
    
    # Obtener datos de la cerveza
    cerveza = recomendar.obtener_cerveza(id_cerveza)
    if not cerveza:
        return redirect(url_for('get_recomendaciones'))
    
    # Obtener reviews existentes
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.rating, r.review_text, r.username, r.date, r.serving_method, r.venue
        FROM ratings_historicos r 
        WHERE r.beer_id = ? AND r.rating > 0
        ORDER BY r.rating DESC, r.date DESC
        LIMIT 10
    """, [id_cerveza])
    reviews = [dict(row) for row in cursor.fetchall()]
    
    # Obtener la calificación del usuario para esta cerveza
    cursor.execute("""
        SELECT rating 
        FROM interaccion 
        WHERE user_id = ? AND beer_id = ?
    """, [user_id, id_cerveza])
    user_rating_row = cursor.fetchone()
    user_rating = int(user_rating_row['rating']) if user_rating_row and user_rating_row['rating'] else 0
    
    # Obtener recomendaciones relacionadas
    id_cervezas, sistema_usado = recomendar.recomendar_contexto(user_id, id_cerveza, N=6)
    cervezas_recomendadas = recomendar.datos_cervezas(id_cervezas)
    
    # Obtener ratings del usuario para las cervezas recomendadas
    user_ratings = {}
    if id_cervezas:
        placeholders = ','.join(['?'] * len(id_cervezas))
        cursor.execute(f"""
            SELECT beer_id, rating 
            FROM interaccion 
            WHERE user_id = ? AND beer_id IN ({placeholders})
        """, [user_id] + id_cervezas)
        for row in cursor.fetchall():
            rating_value = row['rating']
            # Asegurar que convertimos correctamente, incluso si es None o 0
            if rating_value is not None:
                user_ratings[row['beer_id']] = int(float(rating_value))
            else:
                user_ratings[row['beer_id']] = 0
    
    # Estadísticas del usuario
    cant_valoradas = len(recomendar.items_valorados(user_id))
    cant_vistas = len(recomendar.items_vistos(user_id))
    
    conn.close()
    
    return render_template("cerveza_detalle.html", 
                         cerveza=cerveza, 
                         reviews=reviews,
                         cervezas_recomendadas=cervezas_recomendadas, 
                         user_id=user_id, 
                         cant_valoradas=cant_valoradas, 
                         cant_vistas=cant_vistas,
                         user_rating=user_rating,
                         user_ratings=user_ratings)

@app.get('/recomendaciones/<string:id_cerveza>')
def get_recomendaciones_cerveza(id_cerveza):
    user_id = request.cookies.get('user_id')

    id_cervezas, sistema_usado = recomendar.recomendar_contexto(user_id, id_cerveza)

    # pongo cervezas vistas con rating = 0
    for id_cerveza in id_cervezas:
        recomendar.insertar_interacciones(id_cerveza, user_id, 0)

    cervezas_recomendadas = recomendar.datos_cervezas(id_cervezas)
    cant_valoradas = len(recomendar.items_valorados(user_id))
    cant_vistas = len(recomendar.items_vistos(user_id))

    cerveza = recomendar.obtener_cerveza(id_cerveza)

    return render_template("recomendaciones_cerveza.html", cerveza=cerveza, cervezas_recomendadas=cervezas_recomendadas, user_id=user_id, cant_valoradas=cant_valoradas, cant_vistas=cant_vistas)


@app.post('/api/rating')
def post_api_rating():
    """API endpoint para guardar evaluación via AJAX"""
    user_id = request.cookies.get('user_id')
    
    if not user_id:
        return jsonify({"error": "Usuario no autenticado"}), 401
    
    data = request.get_json()
    beer_id = data.get('beer_id')
    rating = data.get('rating')
    
    if not beer_id:
        return jsonify({"error": "beer_id requerido"}), 400
    
    # Validar rating: debe ser entero entre 0 y 5
    try:
        rating = int(rating)
        if rating < 0 or rating > 5:
            return jsonify({"error": "Rating debe estar entre 0 y 5"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Rating debe ser un número entero"}), 400
    
    # Guardar evaluación
    recomendar.insertar_interacciones(beer_id, user_id, rating)
    
    # Verificar y disparar retreinamiento si necesario
    try:
        verificar_y_disparar_retreinamiento(user_id)
    except Exception as e:
        print(f"Error en verificación de retreinamiento: {e}")
    
    return jsonify({"success": True, "beer_id": beer_id, "rating": rating})

@app.post('/recomendaciones')
def post_recomendaciones():
    user_id = request.cookies.get('user_id')

    # inserto los ratings enviados como interacciones
    for id_cerveza in request.form.keys():
        rating = float(request.form[id_cerveza])
        if rating > 0: # 0 es que no puntuó
            recomendar.insertar_interacciones(id_cerveza, user_id, rating)
    
    # Verificar y disparar retreinamiento si necesario
    try:
        verificar_y_disparar_retreinamiento(user_id)
    except Exception as e:
        print(f"Error en verificación de retreinamiento: {e}")

    return make_response(redirect("/recomendaciones"))

@app.get('/buscar')
def get_buscar():
    """Página de búsqueda de cervezas"""
    user_id = request.cookies.get('user_id')
    
    # Debug: verificar user_id
    if not user_id:
        print(f"DEBUG buscar: user_id está None o vacío!")
    else:
        print(f"DEBUG buscar: user_id={user_id}")
    
    query = request.args.get('q', '')
    estilo = request.args.get('estilo', '')
    cerveceria = request.args.get('cerveceria', '')
    abv_min = request.args.get('abv_min', '')
    abv_max = request.args.get('abv_max', '')
    
    cervezas = []
    user_ratings = {}  # Diccionario para almacenar ratings del usuario
    
    if query or estilo or cerveceria or abv_min or abv_max:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Construir query dinámico
        where_conditions = []
        params = []
        
        if query:
            where_conditions.append("(beer_name LIKE ? OR brewery_name LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        
        if estilo:
            where_conditions.append("style = ?")
            params.append(estilo)
            
        if cerveceria:
            where_conditions.append("brewery_name = ?")
            params.append(cerveceria)
            
        if abv_min:
            where_conditions.append("CAST(abv AS REAL) >= ?")
            params.append(float(abv_min))
            
        if abv_max:
            where_conditions.append("CAST(abv AS REAL) <= ?")
            params.append(float(abv_max))
        
        sql = "SELECT * FROM cervezas"
        if where_conditions:
            sql += " WHERE " + " AND ".join(where_conditions)
        sql += " ORDER BY rating DESC, total_ratings DESC LIMIT 50"
        
        cursor.execute(sql, params)
        cervezas = [dict(row) for row in cursor.fetchall()]
        
        # Obtener ratings del usuario para las cervezas encontradas
        if cervezas and user_id:
            beer_ids = [c['beer_id'] for c in cervezas]
            if beer_ids:
                placeholders = ','.join(['?'] * len(beer_ids))
                query_ratings = f"""
                    SELECT beer_id, rating 
                    FROM interaccion 
                    WHERE user_id = ? AND beer_id IN ({placeholders})
                """
                cursor.execute(query_ratings, [user_id] + beer_ids)
                ratings_rows = cursor.fetchall()
                print(f"DEBUG buscar: user_id={user_id}")
                print(f"DEBUG buscar: beer_ids buscados={beer_ids[:5] if len(beer_ids) > 5 else beer_ids}")
                print(f"DEBUG buscar: ratings encontrados en DB={len(ratings_rows)}")
                for row in ratings_rows:
                    rating_value = row['rating']
                    beer_id = str(row['beer_id'])  # Asegurar que beer_id sea string
                    # Asegurar que convertimos correctamente, incluso si es None o 0
                    if rating_value is not None:
                        user_ratings[beer_id] = int(float(rating_value))
                        print(f"DEBUG buscar: beer_id={beer_id}, rating={user_ratings[beer_id]}")
                    else:
                        user_ratings[beer_id] = 0
                print(f"DEBUG buscar: user_ratings final={user_ratings}")
            else:
                print(f"DEBUG buscar: No hay beer_ids para buscar ratings")
        else:
            if not cervezas:
                print(f"DEBUG buscar: No hay cervezas encontradas")
            if not user_id:
                print(f"DEBUG buscar: user_id está None, no se pueden buscar ratings")
        
        # Asegurar que user_ratings siempre sea un diccionario válido
        if not isinstance(user_ratings, dict):
            user_ratings = {}
        
        conn.close()
    
    # Obtener opciones para filtros
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT style FROM cervezas WHERE style IS NOT NULL AND style != '' ORDER BY style")
    estilos = [row['style'] for row in cursor.fetchall()]
    
    cursor.execute("SELECT DISTINCT brewery_name FROM cervezas WHERE brewery_name IS NOT NULL AND brewery_name != '' ORDER BY brewery_name")
    cervecerias = [row['brewery_name'] for row in cursor.fetchall()]
    conn.close()
    
    cant_valoradas = len(recomendar.items_valorados(user_id))
    cant_vistas = len(recomendar.items_vistos(user_id))
    
    return render_template("buscar.html", 
                         cervezas=cervezas,
                         user_ratings=user_ratings,
                         query=query,
                         estilo=estilo,
                         cerveceria=cerveceria,
                         abv_min=abv_min,
                         abv_max=abv_max,
                         estilos=estilos,
                         cervecerias=cervecerias,
                         user_id=user_id,
                         cant_valoradas=cant_valoradas,
                         cant_vistas=cant_vistas)

@app.get('/historial')
def get_historial():
    """Página de historial de evaluaciones del usuario"""
    user_id = request.cookies.get('user_id')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.rating, i.fecha, c.beer_name, c.brewery_name, c.style, c.abv, c.beer_id
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.user_id = ? AND i.rating > 0
        ORDER BY i.fecha DESC
    """, [user_id])
    evaluaciones = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    cant_valoradas = len(recomendar.items_valorados(user_id))
    cant_vistas = len(recomendar.items_vistos(user_id))
    
    return render_template("historial.html", 
                         evaluaciones=evaluaciones,
                         user_id=user_id,
                         cant_valoradas=cant_valoradas,
                         cant_vistas=cant_vistas)

@app.get('/perfil')
def get_perfil():
    """Página de perfil del usuario con estadísticas"""
    user_id = request.cookies.get('user_id')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Estadísticas básicas
    cursor.execute("SELECT COUNT(*) as total FROM interaccion WHERE user_id = ? AND rating > 0", [user_id])
    total_evaluaciones = cursor.fetchone()['total']
    
    cursor.execute("SELECT AVG(rating) as promedio FROM interaccion WHERE user_id = ? AND rating > 0", [user_id])
    promedio_rating = cursor.fetchone()['promedio'] or 0
    
    # Estilos preferidos
    cursor.execute("""
        SELECT c.style, COUNT(*) as cantidad, AVG(i.rating) as promedio
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.user_id = ? AND i.rating > 0 AND c.style IS NOT NULL AND c.style != ''
        GROUP BY c.style
        ORDER BY cantidad DESC, promedio DESC
        LIMIT 5
    """, [user_id])
    estilos_preferidos = [dict(row) for row in cursor.fetchall()]
    
    # Cervecerías preferidas
    cursor.execute("""
        SELECT c.brewery_name, COUNT(*) as cantidad, AVG(i.rating) as promedio
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.user_id = ? AND i.rating > 0 AND c.brewery_name IS NOT NULL AND c.brewery_name != ''
        GROUP BY c.brewery_name
        ORDER BY cantidad DESC, promedio DESC
        LIMIT 5
    """, [user_id])
    cervecerias_preferidas = [dict(row) for row in cursor.fetchall()]
    
    # Cervezas mejor evaluadas
    cursor.execute("""
        SELECT c.beer_name, c.brewery_name, i.rating, i.fecha
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.user_id = ? AND i.rating > 0
        ORDER BY i.rating DESC, i.fecha DESC
        LIMIT 10
    """, [user_id])
    mejores_cervezas = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    cant_valoradas = len(recomendar.items_valorados(user_id))
    cant_vistas = len(recomendar.items_vistos(user_id))
    
    return render_template("perfil.html", 
                         user_id=user_id,
                         total_evaluaciones=total_evaluaciones,
                         promedio_rating=round(promedio_rating, 2),
                         estilos_preferidos=estilos_preferidos,
                         cervecerias_preferidas=cervecerias_preferidas,
                         mejores_cervezas=mejores_cervezas,
                         cant_valoradas=cant_valoradas,
                         cant_vistas=cant_vistas)

@app.get('/admin')
def get_admin():
    """Página de administración para monitorear usuarios"""
    user_id = request.cookies.get('user_id')
    
    # Verificar si es admin - solo rmarques puede acceder
    if user_id != ADMIN_USER_ID:
        return redirect('/recomendaciones')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Estadísticas generales del sistema
    cursor.execute("SELECT COUNT(*) as total FROM usuarios")
    total_usuarios = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as total FROM interaccion WHERE rating > 0")
    total_evaluaciones = cursor.fetchone()['total']
    
    cursor.execute("SELECT COUNT(*) as total FROM interaccion WHERE rating = 0")
    total_vistas = cursor.fetchone()['total']
    
    # Lista de usuarios con sus estadísticas (optimizada)
    cursor.execute("""
        SELECT 
            user_id,
            COUNT(CASE WHEN rating > 0 THEN 1 END) as evaluaciones,
            COUNT(CASE WHEN rating = 0 THEN 1 END) as vistas,
            AVG(CASE WHEN rating > 0 THEN rating END) as rating_promedio,
            MIN(fecha) as primera_actividad,
            MAX(fecha) as ultima_actividad
        FROM interaccion
        GROUP BY user_id
        ORDER BY evaluaciones DESC, vistas DESC
        LIMIT 50
    """)
    usuarios_stats = [dict(row) for row in cursor.fetchall()]
    
    # Top cervezas más evaluadas (optimizada)
    cursor.execute("""
        SELECT 
            c.beer_name,
            c.brewery_name,
            c.style,
            COUNT(*) as total_evaluaciones,
            AVG(i.rating) as rating_promedio
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.rating > 0
        GROUP BY c.beer_id
        ORDER BY total_evaluaciones DESC
        LIMIT 10
    """)
    top_cervezas = [dict(row) for row in cursor.fetchall()]
    
    # Top estilos más populares (optimizada)
    cursor.execute("""
        SELECT 
            c.style,
            COUNT(*) as total_evaluaciones,
            AVG(i.rating) as rating_promedio
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.rating > 0 AND c.style IS NOT NULL AND c.style != ''
        GROUP BY c.style
        ORDER BY total_evaluaciones DESC
        LIMIT 10
    """)
    top_estilos = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return render_template("admin.html", 
                         user_id=user_id,
                         total_usuarios=total_usuarios,
                         total_evaluaciones=total_evaluaciones,
                         total_vistas=total_vistas,
                         usuarios_stats=usuarios_stats,
                         top_cervezas=top_cervezas,
                         top_estilos=top_estilos)

@app.get('/admin/usuario/<string:target_user_id>')
def get_admin_usuario(target_user_id):
    """Detalles de un usuario específico para admin"""
    user_id = request.cookies.get('user_id')
    
    # Verificar si es admin - solo rmarques puede acceder
    if user_id != ADMIN_USER_ID:
        return redirect('/recomendaciones')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Informaciones del usuario
    cursor.execute("""
        SELECT 
            COUNT(CASE WHEN i.rating > 0 THEN 1 END) as evaluaciones,
            COUNT(CASE WHEN i.rating = 0 THEN 1 END) as vistas,
            AVG(CASE WHEN i.rating > 0 THEN i.rating END) as rating_promedio,
            MIN(i.fecha) as primera_actividad,
            MAX(i.fecha) as ultima_actividad
        FROM interaccion i
        WHERE i.user_id = ?
    """, [target_user_id])
    user_stats = cursor.fetchone()
    
    # Evaluaciones del usuario
    cursor.execute("""
        SELECT i.rating, i.fecha, c.beer_name, c.brewery_name, c.style, c.abv
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.user_id = ? AND i.rating > 0
        ORDER BY i.fecha DESC
    """, [target_user_id])
    evaluaciones = [dict(row) for row in cursor.fetchall()]
    
    # Estilos preferidos del usuario
    cursor.execute("""
        SELECT c.style, COUNT(*) as cantidad, AVG(i.rating) as promedio
        FROM interaccion i
        JOIN cervezas c ON i.beer_id = c.beer_id
        WHERE i.user_id = ? AND i.rating > 0 AND c.style IS NOT NULL AND c.style != ''
        GROUP BY c.style
        ORDER BY cantidad DESC, promedio DESC
        LIMIT 5
    """, [target_user_id])
    estilos_preferidos = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return render_template("admin_usuario.html", 
                         user_id=user_id,
                         target_user_id=target_user_id,
                         user_stats=dict(user_stats) if user_stats else {},
                         evaluaciones=evaluaciones,
                         estilos_preferidos=estilos_preferidos)

@app.post('/admin/retrain')
def post_admin_retrain():
    """Forzar retreinamento manual del modelo"""
    user_id = request.cookies.get('user_id')
    
    if user_id != ADMIN_USER_ID:
        return {"error": "No autorizado"}, 403
    
    history_id = trigger_retrain_global("manual_admin")
    
    return {"success": True, "history_id": history_id, "message": "Retreinamento iniciado en background"}

@app.get('/reset')
def get_reset():
    user_id = request.cookies.get('user_id')
    recomendar.reset_usuario(user_id)

    return make_response(redirect("/recomendaciones"))

if __name__ == '__main__':
    app.run(port=5001)


