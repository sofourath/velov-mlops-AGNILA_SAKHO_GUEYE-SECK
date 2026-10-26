"""Simulateur de données Vélo'v (disponibilité horaire par station).

Pourquoi un simulateur ?
- Les sessions S1-S2 portent sur l'industrialisation, pas sur l'accès aux données :
  un jeu reproductible (seed fixe) garantit que tout le monde obtient les mêmes chiffres.
- En S3, ce module est remplacé par une ingestion de l'open data de la Métropole de Lyon,
  avec le même schéma de sortie. C'est le "data contract" du projet.

Schéma de sortie (une ligne par station et par heure) :
    station_id:int, timestamp:datetime64 en UTC, capacity:int, bikes_available:int,
    temperature:float (°C), is_raining:bool

Convention horaire du projet : tous les instants sont stockés et échangés en UTC.
Les usages, eux, suivent l'heure locale de Lyon (départs vers 8 h, retours vers 18 h) :
les profils ci-dessous sont donc calculés en heure locale, puis l'instant est converti en UTC.

Usage :
    python -m velov.data --days 90 --stations 20 --out data/velov_history.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Profils de remplissage (taux d'occupation 0-1 selon l'heure, jours ouvrés).
# residential : se vide le matin (départs au travail), se remplit le soir.
# business    : l'inverse.
# leisure     : peu sensible aux heures de bureau, plus chargée le week-end.
PROFILES = ("residential", "business", "leisure")
BRONZE_KEY_COLUMNS = ("station_id", "timestamp")
BRONZE_COLUMNS = (
    "station_id",
    "timestamp",
    "capacity",
    "bikes_available",
    "temperature",
    "is_raining",
)


def _weekday_occupancy(hour: np.ndarray, profile: str) -> np.ndarray:
    morning = np.exp(-((hour - 8.5) ** 2) / 3.0)
    evening = np.exp(-((hour - 18.0) ** 2) / 4.0)
    if profile == "residential":
        return 0.65 - 0.45 * morning + 0.25 * evening
    if profile == "business":
        return 0.25 + 0.55 * morning - 0.15 * evening
    return 0.45 + 0.10 * np.sin((hour - 6) / 24 * 2 * np.pi)


def _weekend_occupancy(hour: np.ndarray, profile: str) -> np.ndarray:
    afternoon = np.exp(-((hour - 15.0) ** 2) / 8.0)
    if profile == "leisure":
        return 0.40 - 0.30 * afternoon
    return 0.55 - 0.15 * afternoon


def simulate(
    days: int = 90,
    n_stations: int = 20,
    start: str = "2026-06-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Génère un historique horaire synthétique et réaliste.

    Raises:
        ValueError: si les paramètres sont hors bornes.
    """
    if days < 2:
        raise ValueError("days doit être >= 2 (il faut au moins une heure suivante pour la cible)")
    if not 1 <= n_stations <= 500:
        raise ValueError("n_stations doit être compris entre 1 et 500")

    rng = np.random.default_rng(seed)
    # Heure locale pour simuler les usages, UTC pour stocker l'instant.
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h", tz="Europe/Paris")
    hour = np.asarray(timestamps.hour)
    is_weekend = np.asarray(timestamps.dayofweek) >= 5

    # Météo commune à toutes les stations (une seule ville).
    daily_temp = 22 + 6 * np.sin(np.arange(days) / days * np.pi) + rng.normal(0, 2, days)
    temperature = np.repeat(daily_temp, 24) + 5 * np.sin((hour - 9) / 24 * 2 * np.pi)
    is_raining = rng.random(len(timestamps)) < 0.08

    frames = []
    for station_id in range(1, n_stations + 1):
        profile = PROFILES[(station_id - 1) % len(PROFILES)]
        capacity = int(rng.integers(15, 41))
        occ = np.where(
            is_weekend,
            _weekend_occupancy(hour, profile),
            _weekday_occupancy(hour, profile),
        )
        # La pluie réduit les départs : les stations restent plus pleines.
        occ = occ + 0.10 * is_raining
        # Bruit autocorrélé (AR(1)) : la disponibilité d'une heure dépend de la précédente.
        noise = np.zeros(len(timestamps))
        for t in range(1, len(noise)):
            noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 0.05)
        bikes = np.clip(np.rint((occ + noise) * capacity), 0, capacity).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "timestamp": timestamps.tz_convert("UTC"),
                    "capacity": capacity,
                    "bikes_available": bikes,
                    "temperature": temperature.round(1),
                    "is_raining": is_raining,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def ingest_bronze(data: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """Ajoute un lot dans la zone bronze sans créer de doublons.

    Une observation est identifiée par sa station et son horodatage UTC. Les
    observations du lot courant remplacent celles portant la même clé déjà
    présentes, ce qui permet de rejouer un lot ou de corriger une observation
    sans multiplier les lignes.
    """
    missing = set(BRONZE_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour l'ingestion bronze : {sorted(missing)}")

    incoming = data.loc[:, BRONZE_COLUMNS].copy()
    incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)
    if incoming[list(BRONZE_KEY_COLUMNS)].duplicated().any():
        incoming = incoming.drop_duplicates(subset=BRONZE_KEY_COLUMNS, keep="last")

    if output_path.exists():
        existing = pd.read_csv(output_path, parse_dates=["timestamp"])
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        combined = pd.concat([existing.loc[:, BRONZE_COLUMNS], incoming], ignore_index=True)
    else:
        combined = incoming

    bronze = (
        combined.drop_duplicates(subset=BRONZE_KEY_COLUMNS, keep="last")
        .sort_values(list(BRONZE_KEY_COLUMNS))
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    bronze.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return bronze


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère un historique Vélo'v synthétique.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--stations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/velov_history.csv"))
    parser.add_argument(
        "--bronze-out",
        type=Path,
        help="Ajoute le lot généré à ce fichier bronze en dédupliquant station_id/timestamp.",
    )
    args = parser.parse_args()

    df = simulate(days=args.days, n_stations=args.stations, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.bronze_out:
        ingest_bronze(df, args.bronze_out)
    print(f"{len(df):,} lignes écrites dans {args.out}")


if __name__ == "__main__":
    main()
