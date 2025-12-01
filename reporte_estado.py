#!/usr/bin/env python3
"""
Reporte integral del estado del sistema de recomendación.

Permite inspeccionar datos, artefactos de modelo y (opcionalmente) ejecutar una
evaluación resumida para tener métricas al día.
"""

import argparse
import os
import pickle
from datetime import datetime

from config import DATABASE_FILE, MODEL_PATH, MAPPINGS_PATH
from database import get_db_connection


def collect_dataset_summary():
    """Recupera conteos básicos de las tablas relevantes."""
    conn = get_db_connection()
    cursor = conn.cursor()
    tables = ["usuarios", "cervezas", "ratings_historicos", "interaccion"]
    summary = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) as total FROM {table}")
        summary[table] = cursor.fetchone()["total"]
    cursor.execute(
        """
        SELECT AVG(rating) as avg_rating, MIN(rating) as min_rating,
               MAX(rating) as max_rating
        FROM ratings_historicos
        WHERE rating > 0
        """
    )
    rating_stats = cursor.fetchone()
    conn.close()
    return {
        "counts": summary,
        "ratings": {
            "avg": rating_stats["avg_rating"] or 0.0,
            "min": rating_stats["min_rating"] or 0.0,
            "max": rating_stats["max_rating"] or 0.0,
        },
    }


def collect_model_state(sample_users=100):
    """Chequea existencia del modelo y cobertura aproximada."""
    model_exists = os.path.exists(MODEL_PATH)
    mappings_exists = os.path.exists(MAPPINGS_PATH)
    model_info = {
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "model_mtime": datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()
        if model_exists
        else None,
        "mappings_path": MAPPINGS_PATH,
        "mappings_exists": mappings_exists,
        "coverage": None,
    }
    if not mappings_exists:
        return model_info

    with open(MAPPINGS_PATH, "rb") as handler:
        mappings = pickle.load(handler)

    coverage = {
        "n_users": mappings.get("n_users"),
        "n_beers": mappings.get("n_beers"),
        "n_styles": mappings.get("n_styles"),
        "n_breweries": mappings.get("n_breweries"),
        "users_in_model": len(mappings.get("user_to_idx", {})),
        "beers_in_model": len(mappings.get("beer_to_idx", {})),
    }

    coverage.update(sample_two_tower_coverage(sample_users))
    model_info["coverage"] = coverage
    return model_info


def sample_two_tower_coverage(sample_users):
    """Estima cobertura del modelo usando una muestra pequeña."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT username, COUNT(*) as num_ratings
        FROM ratings_historicos
        WHERE rating > 0
        GROUP BY username
        HAVING num_ratings >= 10
        ORDER BY num_ratings DESC
        LIMIT ?
        """,
        [sample_users],
    )
    sample = cursor.fetchall()
    cursor.execute("SELECT DISTINCT beer_id FROM cervezas")
    all_beers = [row["beer_id"] for row in cursor.fetchall()]
    conn.close()

    if not os.path.exists(MAPPINGS_PATH):
        return {}

    with open(MAPPINGS_PATH, "rb") as handler:
        mappings = pickle.load(handler)

    beer_to_idx = mappings.get("beer_to_idx", {})
    user_to_idx = mappings.get("user_to_idx", {})

    users_in_model = sum(1 for row in sample if row["username"] in user_to_idx)
    sample_beers = len(all_beers)
    beers_in_model = sum(1 for beer in all_beers if beer in beer_to_idx)

    return {
        "sampled_users": len(sample),
        "sample_users_in_model": users_in_model,
        "total_catalog_beers": sample_beers,
        "catalog_beers_in_model": beers_in_model,
    }


def maybe_run_evaluation(max_users, k):
    """Ejecuta evaluación rápida si se solicitó."""
    if max_users <= 0:
        return None

    from evaluar import evaluar_estrategias  # Import tardío para evitar ciclos

    return evaluar_estrategias(n_usuarios=max_users, k=k)


def render_report(dataset_summary, model_state, evaluation_results, output_path=None):
    """Renderiza el reporte en texto plano y lo opcionalmente lo persiste."""
    lines = []
    lines.append("=" * 80)
    lines.append("REPORTE DEL SISTEMA DE RECOMENDACIÓN")
    lines.append("=" * 80)
    lines.append(f"Base de datos: {DATABASE_FILE}")
    lines.append(f"Generado: {datetime.now().isoformat()}")
    lines.append("")

    # Estado de datos
    lines.append("📊 DATOS")
    for table, total in dataset_summary["counts"].items():
        lines.append(f" - {table}: {total:,}")
    rating_stats = dataset_summary["ratings"]
    lines.append(
        f" - Rating promedio: {rating_stats['avg']:.2f} "
        f"(min={rating_stats['min']:.1f}, max={rating_stats['max']:.1f})"
    )
    lines.append("")

    # Estado de modelo
    lines.append("🤖 MODELO TWO-TOWER")
    lines.append(f" - Modelo: {model_state['model_path']} => {model_state['model_exists']}")
    if model_state["model_mtime"]:
        lines.append(f"   Última modificación: {model_state['model_mtime']}")
    lines.append(
        f" - Mappings: {model_state['mappings_path']} => {model_state['mappings_exists']}"
    )
    if model_state["coverage"]:
        cov = model_state["coverage"]
        lines.append(
            f"   Usuarios en modelo: {cov['users_in_model']}/{cov['n_users']} "
            f"({cov['sample_users_in_model']}/{cov['sampled_users']} en muestra reciente)"
        )
        lines.append(
            f"   Cervezas en modelo: {cov['beers_in_model']}/{cov['n_beers']} "
            f"({cov['catalog_beers_in_model']}/{cov['total_catalog_beers']} catálogo)"
        )
    lines.append("")

    # Métricas
    lines.append("📈 MÉTRICAS DE EVALUACIÓN")
    if evaluation_results:
        for estrategia, datos in evaluation_results.items():
            lines.append(f" - {estrategia}: usuarios={datos['usuarios']}")
            lines.append(
                f"   Train  NDCG={datos['treino']['ndcg']:.4f} "
                f"Precision={datos['treino']['precision']:.4f} "
                f"Recall={datos['treino']['recall']:.4f}"
            )
            lines.append(
                f"   Test   NDCG={datos['teste']['ndcg']:.4f} "
                f"Precision={datos['teste']['precision']:.4f} "
                f"Recall={datos['teste']['recall']:.4f}"
            )
    else:
        lines.append(" - No se ejecutó evaluación (utilizar --eval-users para habilitarla).")
    lines.append("")

    report = "\n".join(lines)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handler:
            handler.write(report)
    return report


def parse_args():
    """Parsea argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Genera reporte del estado del sistema de recomendación."
    )
    parser.add_argument(
        "--eval-users",
        type=int,
        default=0,
        help="Cantidad de usuarios para ejecutar evaluación (0 = deshabilitado).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Valor de K para las métricas cuando se ejecuta la evaluación.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Archivo donde persistir el reporte (stdout por defecto).",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=100,
        help="Tamaño de muestra para estimar cobertura del modelo.",
    )
    return parser.parse_args()


def main():
    """Punto de entrada CLI."""
    args = parse_args()
    dataset_summary = collect_dataset_summary()
    model_state = collect_model_state(sample_users=args.sample_users)
    evaluation_results = maybe_run_evaluation(args.eval_users, args.k)
    report = render_report(dataset_summary, model_state, evaluation_results, args.output)
    print(report)


if __name__ == "__main__":
    main()

