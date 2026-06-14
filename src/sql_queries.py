"""
sql_queries.py — Base SQLite et requêtes analytiques
CEET Smart Grid · Energy Blackout Prediction Project
"""

import sqlite3
import pandas as pd
from pathlib import Path

from utils import get_logger, DATA_PROC, ROOT_DIR

logger = get_logger("sql")
DB_PATH = ROOT_DIR / "data" / "processed" / "ceet_smartgrid.db"


# ─── Création / chargement de la base ────────────────────────────────────────
def create_database(df: pd.DataFrame = None, db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Crée la base SQLite et charge les données si fourni."""
    conn = sqlite3.connect(db_path)
    logger.info(f"Connexion SQLite : {db_path}")

    if df is not None:
        # Table principale
        df.to_sql("grid_readings", conn, if_exists="replace", index=False)
        logger.info(f"Table 'grid_readings' : {len(df):,} lignes")

        # Création des index pour performance
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE INDEX IF NOT EXISTS idx_datetime ON grid_readings(datetime);
            CREATE INDEX IF NOT EXISTS idx_region   ON grid_readings(region);
            CREATE INDEX IF NOT EXISTS idx_blackout ON grid_readings(blackout);
            CREATE INDEX IF NOT EXISTS idx_overload ON grid_readings(overload);
        """)
        conn.commit()

        # Vue : événements de coupure
        cursor.executescript("""
            DROP VIEW IF EXISTS v_blackout_events;
            CREATE VIEW v_blackout_events AS
            SELECT datetime, region, city, total_load_mw, available_power_mw,
                   outage_risk, load_shedding_mw, event, temperature
            FROM   grid_readings
            WHERE  blackout = 1;
        """)

        # Vue : surcharges
        cursor.executescript("""
            DROP VIEW IF EXISTS v_overload_events;
            CREATE VIEW v_overload_events AS
            SELECT datetime, region, city, total_load_mw, available_power_mw,
                   load_ratio, transformer_temp, voltage, frequency, event
            FROM   grid_readings
            WHERE  overload = 1;
        """)
        conn.commit()
        logger.info("Vues créées : v_blackout_events, v_overload_events")

    return conn


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


# ─── Requêtes analytiques ─────────────────────────────────────────────────────
QUERIES = {

    "overview": """
        SELECT
            COUNT(*)                                           AS total_readings,
            SUM(blackout)                                      AS total_blackouts,
            SUM(overload)                                      AS total_overloads,
            ROUND(AVG(total_load_mw),    2)                    AS avg_load_mw,
            ROUND(AVG(outage_risk),      2)                    AS avg_outage_risk,
            ROUND(AVG(load_shedding_mw), 2)                    AS avg_shedding_mw,
            MIN(datetime)                                      AS period_start,
            MAX(datetime)                                      AS period_end
        FROM grid_readings
    """,

    "blackout_by_region": """
        SELECT
            region,
            COUNT(*)                          AS total,
            SUM(blackout)                     AS blackouts,
            ROUND(100.0 * SUM(blackout) / COUNT(*), 2) AS blackout_rate_pct,
            ROUND(AVG(outage_risk), 2)        AS avg_risk,
            ROUND(AVG(load_shedding_mw), 2)   AS avg_shedding
        FROM   grid_readings
        GROUP  BY region
        ORDER  BY blackout_rate_pct DESC
    """,

    "blackout_by_hour": """
        SELECT
            hour,
            COUNT(*)                          AS total,
            SUM(blackout)                     AS blackouts,
            ROUND(100.0 * SUM(blackout) / COUNT(*), 2) AS blackout_rate_pct,
            ROUND(AVG(total_load_mw), 2)      AS avg_load,
            ROUND(AVG(outage_risk), 2)        AS avg_risk
        FROM   grid_readings
        GROUP  BY hour
        ORDER  BY hour
    """,

    "monthly_summary": """
        SELECT
            strftime('%Y-%m', datetime)            AS month,
            COUNT(*)                               AS readings,
            SUM(blackout)                          AS blackouts,
            SUM(overload)                          AS overloads,
            ROUND(AVG(total_load_mw), 2)           AS avg_load_mw,
            ROUND(AVG(available_power_mw), 2)      AS avg_available_mw,
            ROUND(AVG(outage_risk), 2)             AS avg_risk,
            ROUND(SUM(load_shedding_mw), 2)        AS total_shedding_mw,
            ROUND(AVG(temperature), 2)             AS avg_temp,
            ROUND(AVG(energy_price), 2)            AS avg_price
        FROM   grid_readings
        GROUP  BY strftime('%Y-%m', datetime)
        ORDER  BY month
    """,

    "top_risk_periods": """
        SELECT datetime, region, city, outage_risk, total_load_mw,
               available_power_mw, load_shedding_mw, event, temperature, voltage
        FROM   grid_readings
        WHERE  outage_risk > 90
        ORDER  BY outage_risk DESC
        LIMIT  50
    """,

    "event_impact": """
        SELECT
            event,
            COUNT(*)                          AS occurrences,
            ROUND(AVG(total_load_mw), 2)      AS avg_load_mw,
            ROUND(AVG(outage_risk), 2)        AS avg_risk,
            SUM(blackout)                     AS blackouts,
            ROUND(100.0 * SUM(blackout) / COUNT(*), 2) AS blackout_rate_pct,
            ROUND(AVG(load_shedding_mw), 2)   AS avg_shedding
        FROM   grid_readings
        GROUP  BY event
        ORDER  BY avg_risk DESC
    """,

    "peak_load_analysis": """
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY region ORDER BY total_load_mw DESC) AS rn
            FROM grid_readings
        )
        SELECT region, city, datetime, total_load_mw, available_power_mw,
               outage_risk, event, temperature
        FROM   ranked
        WHERE  rn <= 5
        ORDER  BY region, total_load_mw DESC
    """,

    "power_deficit_windows": """
        SELECT
            datetime, region, city,
            total_load_mw, available_power_mw,
            ROUND(total_load_mw - available_power_mw, 2) AS deficit_mw,
            outage_risk, blackout, load_shedding_mw
        FROM   grid_readings
        WHERE  total_load_mw > available_power_mw
        ORDER  BY deficit_mw DESC
        LIMIT  100
    """,

    "transformer_stress": """
        SELECT
            region, city,
            ROUND(AVG(transformer_temp), 2)  AS avg_temp,
            MAX(transformer_temp)            AS max_temp,
            ROUND(AVG(voltage), 2)           AS avg_voltage,
            ROUND(AVG(frequency), 2)         AS avg_frequency,
            SUM(overload)                    AS overloads,
            SUM(blackout)                    AS blackouts
        FROM   grid_readings
        GROUP  BY region, city
        ORDER  BY avg_temp DESC
    """,

    "season_analysis": """
        SELECT
            season,
            COUNT(*)                              AS readings,
            ROUND(AVG(total_load_mw), 2)          AS avg_load,
            ROUND(AVG(temperature), 2)            AS avg_temp,
            ROUND(AVG(humidity), 2)               AS avg_humidity,
            SUM(blackout)                         AS blackouts,
            SUM(overload)                         AS overloads,
            ROUND(AVG(renewable_power_mw), 2)     AS avg_renewable_mw,
            ROUND(AVG(energy_price), 2)           AS avg_price
        FROM   grid_readings
        GROUP  BY season
        ORDER  BY avg_load DESC
    """,

    "city_risk_ranking": """
        SELECT
            city, region,
            COUNT(*)                              AS total,
            SUM(blackout)                         AS blackouts,
            SUM(overload)                         AS overloads,
            ROUND(AVG(outage_risk), 2)            AS avg_risk,
            ROUND(MAX(outage_risk), 2)            AS max_risk,
            ROUND(AVG(load_shedding_mw), 2)       AS avg_shedding,
            ROUND(AVG(population_density), 0)     AS avg_pop_density
        FROM   grid_readings
        GROUP  BY city, region
        ORDER  BY avg_risk DESC
    """,

    "hourly_load_window": """
        SELECT
            hour,
            ROUND(AVG(total_load_mw), 2)                    AS avg_load,
            ROUND(AVG(total_load_mw) OVER (
                ORDER BY hour ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
            ), 2)                                           AS smoothed_load,
            ROUND(MAX(total_load_mw), 2)                    AS max_load,
            ROUND(MIN(total_load_mw), 2)                    AS min_load,
            SUM(blackout)                                   AS blackouts
        FROM   grid_readings
        GROUP  BY hour
        ORDER  BY hour
    """,

    "renewable_contribution": """
        SELECT
            strftime('%Y-%m', datetime)                   AS month,
            region,
            ROUND(AVG(renewable_power_mw), 2)             AS avg_renewable,
            ROUND(AVG(available_power_mw), 2)             AS avg_available,
            ROUND(100.0 * AVG(renewable_power_mw) /
                  NULLIF(AVG(available_power_mw), 0), 2)  AS renewable_pct
        FROM   grid_readings
        GROUP  BY month, region
        ORDER  BY month, renewable_pct DESC
    """,
}


def run_query(query_name: str, conn: sqlite3.Connection = None) -> pd.DataFrame:
    """Exécute une requête nommée et retourne un DataFrame."""
    if query_name not in QUERIES:
        raise ValueError(f"Requête inconnue: {query_name}. Disponibles: {list(QUERIES.keys())}")
    if conn is None:
        conn = get_connection()
    sql = QUERIES[query_name]
    df = pd.read_sql_query(sql, conn)
    logger.info(f"Requête '{query_name}' → {len(df)} lignes")
    return df


def run_all_queries(conn: sqlite3.Connection = None) -> dict:
    """Exécute toutes les requêtes analytiques."""
    if conn is None:
        conn = get_connection()
    results = {}
    for name in QUERIES:
        try:
            results[name] = run_query(name, conn)
        except Exception as e:
            logger.error(f"Erreur requête '{name}': {e}")
    return results


def custom_query(sql: str, conn: sqlite3.Connection = None) -> pd.DataFrame:
    """Exécute une requête SQL personnalisée."""
    if conn is None:
        conn = get_connection()
    return pd.read_sql_query(sql, conn)


if __name__ == "__main__":
    # Test rapide
    proc_csv = DATA_PROC / "ceet_processed.csv"
    if proc_csv.exists():
        df = pd.read_csv(proc_csv, nrows=5000)
        conn = create_database(df)
        results = run_all_queries(conn)
        for name, res in results.items():
            print(f"\n{'='*50}")
            print(f"  {name.upper()}")
            print(res.to_string(index=False))
    else:
        print("Dataset traité non trouvé. Lancez d'abord data_preprocessing.py")
