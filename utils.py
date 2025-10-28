#!/usr/bin/env python3
"""
Utilidades y Métricas del Sistema
Consolidado desde metricas.py, resumen_rapido.py, resumen_db.py
"""

import sqlite3
import pandas as pd
import math
import os
from datetime import datetime
from config import DATABASE_FILE
from database import get_db_connection

# =============================================================================
# SECCIÓN: MÉTRICAS DE RECOMENDACIÓN
# =============================================================================

def discounted_cumulative_gain(relevance_scores):
    """Calcula Discounted Cumulative Gain (DCG)"""
    if not relevance_scores:
        return 0.0

    dcg = 0.0
    for i, relevance in enumerate(relevance_scores):
        dcg += relevance / math.log2(i + 1 + 1)
    return dcg

def ideal_discounted_cumulative_gain(relevance_scores):
    """Calcula Ideal Discounted Cumulative Gain (IDCG)"""
    sorted_relevance = sorted(relevance_scores, reverse=True)
    return discounted_cumulative_gain(sorted_relevance)

def normalized_discounted_cumulative_gain(relevance_scores):
    """Calcula Normalized Discounted Cumulative Gain (NDCG)"""
    dcg = discounted_cumulative_gain(relevance_scores)
    idcg = ideal_discounted_cumulative_gain(relevance_scores)

    if idcg == 0:
        return 0.0

    return dcg / idcg

def precision_at_k(recommended_items, relevant_items, k):
    """Calcula Precision@K"""
    if k == 0:
        return 0.0
    
    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant_items))
    return relevant_recommended / k

def recall_at_k(recommended_items, relevant_items, k):
    """Calcula Recall@K"""
    if len(relevant_items) == 0:
        return 0.0
    
    recommended_k = recommended_items[:k]
    relevant_recommended = len(set(recommended_k) & set(relevant_items))
    return relevant_recommended / len(relevant_items)

def f1_at_k(recommended_items, relevant_items, k):
    """Calcula F1@K"""
    precision = precision_at_k(recommended_items, relevant_items, k)
    recall = recall_at_k(recommended_items, relevant_items, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

# =============================================================================
# SECCIÓN: RESUMEN RÁPIDO DE BASE DE DATOS
# =============================================================================

def resumen_rapido(db_path=None):
    """Genera un resumen rápido de la base de datos"""
    if db_path is None:
        db_path = DATABASE_FILE
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("📊 RESUMEN RÁPIDO - BASE DE DATOS UNTAPPD")
        print("=" * 50)
        
        # Conteos básicos
        tables = ['usuarios', 'cervezas', 'cervecerias', 'interaccion', 'ratings_historicos']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"🔹 {table.upper()}: {count:,} registros")
        
        # Métricas clave
        print("\n📈 MÉTRICAS CLAVE:")
        
        # Usuarios activos
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM ratings_historicos;")
        active_users = cursor.fetchone()[0]
        print(f"   • Usuarios activos: {active_users:,}")
        
        # Cervezas con ratings
        cursor.execute("SELECT COUNT(DISTINCT beer_id) FROM ratings_historicos;")
        rated_beers = cursor.fetchone()[0]
        print(f"   • Cervezas con ratings: {rated_beers:,}")
        
        # Rating promedio
        cursor.execute("SELECT AVG(rating) FROM ratings_historicos;")
        avg_rating = cursor.fetchone()[0]
        print(f"   • Rating promedio: {avg_rating:.2f}")
        
        # Densidad de interacciones
        cursor.execute("SELECT COUNT(*) FROM interaccion;")
        interactions = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM usuarios;")
        total_users = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM cervezas;")
        total_beers = cursor.fetchone()[0]
        
        if total_users > 0 and total_beers > 0:
            max_possible = total_users * total_beers
            density = (interactions / max_possible) * 100
            print(f"   • Densidad matriz: {density:.6f}%")
        
        # Top estilo
        cursor.execute("""
            SELECT style, COUNT(*) as cantidad 
            FROM cervezas 
            WHERE style IS NOT NULL 
            GROUP BY style 
            ORDER BY cantidad DESC 
            LIMIT 1;
        """)
        top_style = cursor.fetchone()
        if top_style:
            print(f"   • Estilo más popular: {top_style[0]} ({top_style[1]:,} cervezas)")
        
        conn.close()
        print("\n✓ Resumen completado")
        
    except sqlite3.Error as e:
        print(f"✗ Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error inesperado: {e}")
        return False
    
    return True

# =============================================================================
# SECCIÓN: ANÁLISIS DETALLADO DE BASE DE DATOS
# =============================================================================

class DatabaseAnalyzer:
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_FILE
        self.conn = None
        
    def connect(self):
        """Establece conexión con la base de datos"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✓ Conectado a la base de datos: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"✗ Error conectando a la base de datos: {e}")
            return False
    
    def get_table_info(self):
        """Obtiene información de estructura de todas las tablas"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            table_info[table_name] = columns
            
        return table_info
    
    def get_table_counts(self):
        """Obtiene conteos de registros por tabla"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        counts = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            count = cursor.fetchone()[0]
            counts[table_name] = count
            
        return counts
    
    def analyze_usuarios(self):
        """Análisis detallado de la tabla usuarios"""
        query = """
        SELECT 
            COUNT(*) as total_usuarios,
            AVG(total_ratings) as avg_ratings_por_usuario,
            MIN(total_ratings) as min_ratings,
            MAX(total_ratings) as max_ratings,
            AVG(unique_beers) as avg_cervezas_unicas,
            AVG(unique_breweries) as avg_cervecerias_unicas,
            AVG(avg_rating) as avg_rating_promedio,
            COUNT(DISTINCT preferred_serving) as metodos_servicio_unicos
        FROM usuarios;
        """
        
        df = pd.read_sql_query(query, self.conn)
        return df.iloc[0].to_dict()
    
    def analyze_cervezas(self):
        """Análisis detallado de la tabla cervezas"""
        # Estadísticas generales
        stats_query = """
        SELECT 
            COUNT(*) as total_cervezas,
            COUNT(DISTINCT brewery_id) as cervecerias_unicas,
            COUNT(DISTINCT style) as estilos_unicos,
            AVG(abv) as avg_abv,
            MIN(abv) as min_abv,
            MAX(abv) as max_abv,
            AVG(ibu) as avg_ibu,
            AVG(rating) as avg_rating,
            AVG(total_ratings) as avg_total_ratings
        FROM cervezas;
        """
        
        # Top estilos
        styles_query = """
        SELECT style, COUNT(*) as cantidad
        FROM cervezas 
        WHERE style IS NOT NULL
        GROUP BY style 
        ORDER BY cantidad DESC 
        LIMIT 10;
        """
        
        # Top cervecerías por cantidad de cervezas
        breweries_query = """
        SELECT brewery_name, COUNT(*) as cantidad_cervezas, AVG(rating) as avg_rating
        FROM cervezas 
        GROUP BY brewery_name 
        ORDER BY cantidad_cervezas DESC 
        LIMIT 10;
        """
        
        stats = pd.read_sql_query(stats_query, self.conn).iloc[0].to_dict()
        top_styles = pd.read_sql_query(styles_query, self.conn)
        top_breweries = pd.read_sql_query(breweries_query, self.conn)
        
        return {
            'estadisticas': stats,
            'top_estilos': top_styles,
            'top_cervecerias': top_breweries
        }
    
    def analyze_interacciones(self):
        """Análisis detallado de la tabla interacciones"""
        query = """
        SELECT 
            COUNT(*) as total_interacciones,
            COUNT(DISTINCT user_id) as usuarios_unicos,
            COUNT(DISTINCT beer_id) as cervezas_unicas,
            AVG(rating) as avg_rating,
            MIN(rating) as min_rating,
            MAX(rating) as max_rating,
            COUNT(CASE WHEN rating >= 4.0 THEN 1 END) as ratings_altos,
            COUNT(CASE WHEN rating <= 2.0 THEN 1 END) as ratings_bajos
        FROM interaccion;
        """
        
        df = pd.read_sql_query(query, self.conn)
        return df.iloc[0].to_dict()
    
    def get_data_quality_metrics(self):
        """Métricas de calidad de datos"""
        quality_metrics = {}
        
        # Verificar valores nulos en tablas principales
        tables_to_check = ['usuarios', 'cervezas', 'cervecerias', 'interaccion']
        
        for table in tables_to_check:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({table});")
            columns = cursor.fetchall()
            
            null_counts = {}
            for col in columns:
                col_name = col[1]
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {col_name} IS NULL;")
                null_count = cursor.fetchone()[0]
                null_counts[col_name] = null_count
            
            quality_metrics[table] = null_counts
        
        return quality_metrics
    
    def generate_summary_report(self):
        """Genera el reporte completo de resumen"""
        if not self.connect():
            return None
        
        print("=" * 80)
        print("RESUMEN DE BASE DE DATOS UNTAPPD")
        print("=" * 80)
        print(f"Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base de datos: {self.db_path}")
        print()
        
        # Información de estructura
        print("📊 ESTRUCTURA DE LA BASE DE DATOS")
        print("-" * 50)
        table_info = self.get_table_info()
        table_counts = self.get_table_counts()
        
        for table_name, columns in table_info.items():
            count = table_counts.get(table_name, 0)
            print(f"\n🔹 {table_name.upper()} ({count:,} registros)")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, pk = col
                null_info = "NOT NULL" if not_null else "NULL"
                pk_info = " (PK)" if pk else ""
                print(f"   • {col_name}: {col_type} {null_info}{pk_info}")
        
        # Análisis de usuarios
        print("\n\n👥 ANÁLISIS DE USUARIOS")
        print("-" * 50)
        usuarios_stats = self.analyze_usuarios()
        for key, value in usuarios_stats.items():
            if isinstance(value, float):
                print(f"   • {key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                print(f"   • {key.replace('_', ' ').title()}: {value:,}")
        
        # Análisis de cervezas
        print("\n\n🍺 ANÁLISIS DE CERVEZAS")
        print("-" * 50)
        cervezas_analysis = self.analyze_cervezas()
        stats = cervezas_analysis['estadisticas']
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   • {key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                print(f"   • {key.replace('_', ' ').title()}: {value:,}")
        
        print("\n   📈 TOP 10 ESTILOS DE CERVEZA:")
        if 'top_estilos' in cervezas_analysis and not cervezas_analysis['top_estilos'].empty:
            for _, row in cervezas_analysis['top_estilos'].iterrows():
                print(f"      • {row['style']}: {row['cantidad']:,} cervezas")
        else:
            print("      • No hay datos disponibles")
        
        # Análisis de interacciones
        print("\n\n💫 ANÁLISIS DE INTERACCIONES")
        print("-" * 50)
        interacciones_stats = self.analyze_interacciones()
        for key, value in interacciones_stats.items():
            if isinstance(value, float):
                print(f"   • {key.replace('_', ' ').title()}: {value:,.2f}")
            else:
                print(f"   • {key.replace('_', ' ').title()}: {value:,}")
        
        # Métricas de calidad
        print("\n\n🔍 MÉTRICAS DE CALIDAD DE DATOS")
        print("-" * 50)
        quality_metrics = self.get_data_quality_metrics()
        
        for table, null_counts in quality_metrics.items():
            print(f"\n   📋 {table.upper()}:")
            total_records = table_counts.get(table, 0)
            for col_name, null_count in null_counts.items():
                if null_count > 0:
                    percentage = (null_count / total_records) * 100 if total_records > 0 else 0
                    print(f"      • {col_name}: {null_count:,} nulos ({percentage:.1f}%)")
        
        # Resumen ejecutivo
        print("\n\n📋 RESUMEN EJECUTIVO")
        print("-" * 50)
        total_records = sum(table_counts.values())
        print(f"   • Total de registros en la base: {total_records:,}")
        print(f"   • Número de tablas: {len(table_counts)}")
        print(f"   • Usuarios activos: {table_counts.get('usuarios', 0):,}")
        print(f"   • Cervezas catalogadas: {table_counts.get('cervezas', 0):,}")
        print(f"   • Cervecerías registradas: {table_counts.get('cervecerias', 0):,}")
        print(f"   • Interacciones para recomendación: {table_counts.get('interaccion', 0):,}")
        print(f"   • Ratings históricos: {table_counts.get('ratings_historicos', 0):,}")
        
        # Densidad de la matriz usuario-item
        usuarios_count = table_counts.get('usuarios', 0)
        cervezas_count = table_counts.get('cervezas', 0)
        interacciones_count = table_counts.get('interaccion', 0)
        
        if usuarios_count > 0 and cervezas_count > 0:
            max_possible_interactions = usuarios_count * cervezas_count
            density = (interacciones_count / max_possible_interactions) * 100
            print(f"   • Densidad de matriz usuario-item: {density:.6f}%")
        
        print("\n" + "=" * 80)
        
        self.conn.close()
        return True

# =============================================================================
# SECCIÓN: FUNCIONES AUXILIARES GENERALES
# =============================================================================

def formatear_numero(numero):
    """Formatea números con separadores de miles"""
    return f"{numero:,}"

def formatear_porcentaje(valor, total):
    """Formatea porcentajes"""
    if total == 0:
        return "0.0%"
    return f"{(valor/total)*100:.1f}%"

def obtener_estadisticas_usuario(user_id):
    """Obtiene estadísticas específicas de un usuario"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Estadísticas básicas
    cursor.execute("""
        SELECT 
            COUNT(*) as total_evaluaciones,
            AVG(rating) as rating_promedio,
            MIN(rating) as rating_minimo,
            MAX(rating) as rating_maximo,
            COUNT(DISTINCT beer_id) as cervezas_unicas
        FROM interaccion 
        WHERE user_id = ? AND rating > 0
    """, [user_id])
    
    stats = cursor.fetchone()
    conn.close()
    
    return dict(stats) if stats else {}

def obtener_estadisticas_cerveza(beer_id):
    """Obtiene estadísticas específicas de una cerveza"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Estadísticas básicas
    cursor.execute("""
        SELECT 
            COUNT(*) as total_evaluaciones,
            AVG(rating) as rating_promedio,
            MIN(rating) as rating_minimo,
            MAX(rating) as rating_maximo,
            COUNT(DISTINCT user_id) as usuarios_unicos
        FROM interaccion 
        WHERE beer_id = ? AND rating > 0
    """, [beer_id])
    
    stats = cursor.fetchone()
    conn.close()
    
    return dict(stats) if stats else {}

# =============================================================================
# SECCIÓN: FUNCIONES PRINCIPALES
# =============================================================================

def main():
    """Función principal para análisis completo"""
    analyzer = DatabaseAnalyzer()
    success = analyzer.generate_summary_report()
    
    if success:
        print("✓ Análisis completado exitosamente")
    else:
        print("✗ Error durante el análisis")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "rapido":
            resumen_rapido()
        elif sys.argv[1] == "completo":
            main()
        else:
            print("Uso: python utils.py [rapido|completo]")
    else:
        resumen_rapido()
