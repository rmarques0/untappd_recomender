#!/usr/bin/env python3
"""
Gestión de base de datos SQLite
Consolidado desde migrate_to_sqlite.py
"""

import sqlite3
import pandas as pd
import os
from config import (
    DATABASE_FILE, BREWERIES_CSV, BEERS_CSV, 
    USERS_CSV, RATINGS_CSV, CSV_DIR
)

def get_db_connection():
    """Conecta a la base de datos SQLite"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def crear_schema():
    """Crea el schema de la base de datos SQLite"""
    
    schema_sql = """
    -- Tabla de cervecerías
    CREATE TABLE IF NOT EXISTS cervecerias (
        brewery_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT,
        type TEXT,
        rating REAL,
        total_ratings INTEGER,
        beer_count INTEGER,
        url TEXT
    );

    -- Tabla de cervezas
    CREATE TABLE IF NOT EXISTS cervezas (
        beer_id TEXT PRIMARY KEY,
        beer_name TEXT NOT NULL,
        brewery_id TEXT,
        brewery_name TEXT,
        style TEXT,
        abv REAL,
        ibu INTEGER,
        rating REAL,
        total_ratings INTEGER,
        url TEXT,
        image_url TEXT,
        FOREIGN KEY (brewery_id) REFERENCES cervecerias(brewery_id)
    );

    -- Tabla de usuarios (del sistema de recomendación)
    CREATE TABLE IF NOT EXISTS usuarios (
        user_id TEXT PRIMARY KEY,
        username TEXT,
        total_ratings INTEGER,
        unique_beers INTEGER,
        unique_breweries INTEGER,
        avg_rating REAL,
        preferred_serving TEXT,
        total_venues INTEGER,
        first_rating_date TEXT,
        last_rating_date TEXT,
        activity_span_days INTEGER
    );

    -- Tabla de ratings históricos (del Untappd)
    CREATE TABLE IF NOT EXISTS ratings_historicos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        beer_id TEXT,
        beer_name TEXT,
        user_id TEXT,
        username TEXT,
        rating REAL,
        review_text TEXT,
        serving_method TEXT,
        venue TEXT,
        date TEXT,
        FOREIGN KEY (beer_id) REFERENCES cervezas(beer_id),
        FOREIGN KEY (user_id) REFERENCES usuarios(user_id)
    );

    -- Tabla de interacciones del sistema de recomendación
    CREATE TABLE IF NOT EXISTS interaccion (
        user_id TEXT,
        beer_id TEXT,
        rating REAL,
        fecha TEXT,
        PRIMARY KEY (user_id, beer_id),
        FOREIGN KEY (user_id) REFERENCES usuarios(user_id),
        FOREIGN KEY (beer_id) REFERENCES cervezas(beer_id)
    );

    -- Tabla de control de retreinamiento por usuario
    CREATE TABLE IF NOT EXISTS model_updates (
        user_id TEXT PRIMARY KEY,
        evaluaciones_at_last_update INTEGER DEFAULT 0,
        last_update_date TEXT,
        status TEXT DEFAULT 'completed',
        FOREIGN KEY (user_id) REFERENCES usuarios(user_id)
    );

    -- Tabla de histórico de retreinamentos globais
    CREATE TABLE IF NOT EXISTS retrain_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        retrain_type TEXT NOT NULL,
        trigger_reason TEXT,
        evaluaciones_count INTEGER,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        status TEXT DEFAULT 'running',
        error_message TEXT
    );

    -- Tabla de configuración de retreinamiento
    CREATE TABLE IF NOT EXISTS retrain_config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    -- Inicializar configuración
    INSERT OR IGNORE INTO retrain_config (key, value, updated_at) 
    VALUES ('evaluaciones_desde_ultimo_retrain', '0', datetime('now'));

    INSERT OR IGNORE INTO retrain_config (key, value, updated_at)
    VALUES ('ultimo_retrain_global', datetime('now'), datetime('now'));

    -- Índices para mejorar performance
    CREATE INDEX IF NOT EXISTS idx_cervezas_brewery_id ON cervezas(brewery_id);
    CREATE INDEX IF NOT EXISTS idx_cervezas_style ON cervezas(style);
    CREATE INDEX IF NOT EXISTS idx_cervezas_rating ON cervezas(rating);
    CREATE INDEX IF NOT EXISTS idx_ratings_historicos_beer_id ON ratings_historicos(beer_id);
    CREATE INDEX IF NOT EXISTS idx_ratings_historicos_user_id ON ratings_historicos(user_id);
    CREATE INDEX IF NOT EXISTS idx_ratings_historicos_rating ON ratings_historicos(rating);
    CREATE INDEX IF NOT EXISTS idx_interaccion_user_id ON interaccion(user_id);
    CREATE INDEX IF NOT EXISTS idx_interaccion_beer_id ON interaccion(beer_id);
    """
    
    return schema_sql

def migrate_breweries():
    """Migra datos de cervecerías"""
    print("Migrando cervecerías...")
    
    if not os.path.exists(BREWERIES_CSV):
        print(f"⚠️  Archivo no encontrado: {BREWERIES_CSV}")
        return 0
    
    df = pd.read_csv(BREWERIES_CSV)
    df = df.fillna('')
    
    conn = sqlite3.connect(DATABASE_FILE)
    df.to_sql('cervecerias', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ {len(df)} cervecerías migradas")
    return len(df)

def migrate_beers():
    """Migra datos de cervezas"""
    print("Migrando cervezas...")
    
    if not os.path.exists(BEERS_CSV):
        print(f"⚠️  Archivo no encontrado: {BEERS_CSV}")
        return 0
    
    df = pd.read_csv(BEERS_CSV)
    df = df.fillna('')
    
    # Asegurar que image_url existe
    if 'image_url' not in df.columns:
        df['image_url'] = ''
    
    conn = sqlite3.connect(DATABASE_FILE)
    df.to_sql('cervezas', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ {len(df)} cervezas migradas")
    return len(df)

def migrate_users():
    """Migra datos de usuarios"""
    print("Migrando usuarios...")
    
    if not os.path.exists(USERS_CSV):
        print(f"⚠️  Archivo no encontrado: {USERS_CSV}")
        return 0
    
    df = pd.read_csv(USERS_CSV)
    df = df.fillna('')
    
    conn = sqlite3.connect(DATABASE_FILE)
    df.to_sql('usuarios', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ {len(df)} usuarios migrados")
    return len(df)

def migrate_ratings():
    """Migra ratings históricos"""
    print("Migrando ratings históricos...")
    
    if not os.path.exists(RATINGS_CSV):
        print(f"⚠️  Archivo no encontrado: {RATINGS_CSV}")
        return 0
    
    df = pd.read_csv(RATINGS_CSV)
    df = df.fillna('')
    
    conn = sqlite3.connect(DATABASE_FILE)
    df.to_sql('ratings_historicos', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ {len(df)} ratings históricos migrados")
    return len(df)

def crear_base_datos():
    """Crea la base de datos y ejecuta la migración"""
    
    print("Iniciando migración de CSVs a SQLite...")
    print(f"📁 Directorio de datos: {CSV_DIR}")
    print(f"🗄️  Base de datos: {DATABASE_FILE}")
    print()
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    
    # Crear conexión y schema
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Ejecutar schema
    schema_sql = crear_schema()
    cursor.executescript(schema_sql)
    conn.close()
    
    print("✅ Schema de base de datos creado")
    print()
    
    # Migrar datos
    total_breweries = migrate_breweries()
    total_beers = migrate_beers()
    total_users = migrate_users()
    total_ratings = migrate_ratings()
    
    print()
    print("📊 Resumen de migración:")
    print(f"   🏭 Cervecerías: {total_breweries}")
    print(f"   🍺 Cervezas: {total_beers}")
    print(f"   👤 Usuarios: {total_users}")
    print(f"   ⭐ Ratings: {total_ratings}")
    print()
    
    # Verificar integridad
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM cervecerias")
    breweries_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cervezas")
    beers_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM usuarios")
    users_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM ratings_historicos")
    ratings_count = cursor.fetchone()[0]
    
    conn.close()
    
    print("🔍 Verificación de integridad:")
    print(f"   🏭 Cervecerías en DB: {breweries_count}")
    print(f"   🍺 Cervezas en DB: {beers_count}")
    print(f"   👤 Usuarios en DB: {users_count}")
    print(f"   ⭐ Ratings en DB: {ratings_count}")
    print()
    
    if (breweries_count == total_breweries and 
        beers_count == total_beers and 
        users_count == total_users and 
        ratings_count == total_ratings):
        print("🎉 ¡Migración completada exitosamente!")
        return True
    else:
        print("⚠️  Hay discrepancias en los conteos. Revisar migración.")
        return False

def mostrar_datos_muestra():
    """Muestra datos de muestra de la base de datos"""
    print("\n🔍 Datos de muestra:")
    
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Cervecerías
    cursor.execute("SELECT name, location, type FROM cervecerias LIMIT 3")
    breweries = cursor.fetchall()
    print("\n🏭 Cervecerías:")
    for brewery in breweries:
        print(f"   - {brewery[0]} ({brewery[1]}) - {brewery[2]}")
    
    # Cervezas
    cursor.execute("SELECT beer_name, brewery_name, style, abv FROM cervezas LIMIT 3")
    beers = cursor.fetchall()
    print("\n🍺 Cervezas:")
    for beer in beers:
        print(f"   - {beer[0]} ({beer[1]}) - {beer[2]} - ABV: {beer[3]}%")
    
    # Usuarios
    cursor.execute("SELECT username, total_ratings, avg_rating FROM usuarios LIMIT 3")
    users = cursor.fetchall()
    print("\n👤 Usuarios:")
    for user in users:
        print(f"   - {user[0]} - {user[1]} ratings - Avg: {user[2]}")
    
    conn.close()

def main():
    """Función principal para ejecutar migración"""
    try:
        success = crear_base_datos()
        if success:
            mostrar_datos_muestra()
            print(f"\n✅ Base de datos creada en: {DATABASE_FILE}")
            print("🚀 ¡Listo para usar el sistema de recomendación!")
        else:
            print("\n❌ Error en la migración")
            return False
            
    except Exception as e:
        print(f"\n❌ Error durante la migración: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
