# velov-mlops

Projet fil rouge du cours **Industrialisation de l'IA dans le Cloud** (M2 Data Engineering / IA).

Objectif métier : prédire, pour chaque station Vélo'v, le nombre de vélos disponibles **dans une heure**,
et servir cette prédiction via une API fiable, conteneurisée, puis déployée (on-premise et cloud).

## Progression par session

Chaque étape du cours correspond à un tag git. Absent ou bloqué : repartez du tag de fin de la session précédente.

| Tag | Contenu |
|---|---|
| `s1-start` | Données simulées, features, entraînement. API à compléter (TP1) |
| `s1-end` | Modèle v0 servi par FastAPI, tests pytest |

```bash
git checkout s1-end          # récupérer l'état de fin de S1
git checkout -b mon-binome   # travailler sur sa propre branche
```

## Démarrage rapide

Prérequis : Python 3.11+ (3.12 recommandé), git. Docker sera nécessaire en S2.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .                     # rend le package velov importable

python -m velov.data                 # génère data/velov_history.csv (simulateur, seed fixe)
python -m velov.data --bronze-out data/bronze.csv  # ingestion bronze rejouable, sans doublons
dbt build --project-dir dbt --profiles-dir dbt  # modèles silver/gold et contrat gold
dbt seed --project-dir dbt --profiles-dir dbt  # charge la bronze dans DuckDB
dbt run --project-dir dbt --profiles-dir dbt   # construit silver puis gold
python -m velov.train                # lit main.velov_gold, valide Pandera, puis entraîne
python -m velov.train --mlflow       # idem + journalisation MLflow (puis : mlflow ui)
pytest                               # tests
uvicorn velov.api.main:app --reload  # API sur http://127.0.0.1:8000/docs (--reload : en dev uniquement)
```

### Tester Docker Compose

Prérequis : Docker Desktop démarré et `POSTGRES_PASSWORD` défini dans `.env`.
La stack applicative démarre l'API sur `8000` et PostgreSQL sur le réseau interne :

```powershell
docker compose up -d --build
docker compose ps
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/ready
```

Le résultat attendu est `{"status":"ok"}` pour `/health` et un statut HTTP
`200` avec `model_version` pour `/ready`. Les logs se consultent avec :

```powershell
docker compose logs -f api
```

Pour tester aussi MLflow, lancer les deux fichiers Compose ensemble :

```powershell
docker compose -f compose.yaml -f compose.mlflow.yaml up -d --build
docker compose -f compose.yaml -f compose.mlflow.yaml ps
curl.exe http://127.0.0.1:5000/version
```

L'interface MLflow est alors disponible sur
`http://127.0.0.1:5000`. Pour suivre les logs :

```powershell
docker compose -f compose.yaml -f compose.mlflow.yaml logs -f mlflow
```

Arrêter les services sans supprimer les données :

```powershell
docker compose -f compose.yaml -f compose.mlflow.yaml down
```

Pour repartir de zéro, y compris les bases PostgreSQL et les artefacts MLflow,
utiliser `down -v` :

```powershell
docker compose -f compose.yaml -f compose.mlflow.yaml down -v
```

Exemple d'appel :

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"station_id": 3, "timestamp": "2026-10-06T08:00:00+02:00", "capacity": 20,
       "bikes_available": 12, "temperature": 14.5, "is_raining": false}'
```

Le `timestamp` doit porter un fuseau (`+02:00`, `Z`...) : sans fuseau, l'API répond 422.
Les instants sont renvoyés et journalisés en UTC (`"target_timestamp": "2026-10-06T07:00:00Z"`).

## Exigences du projet

Le service doit respecter les exigences de [docs/exigences.md](docs/exigences.md), de S1 à S9.
Chaque rendu indique celles qu'il couvre et comment le vérifier.


## Structure

```
src/velov/
  data.py          simulateur et ingestion bronze idempotente (clé station_id/timestamp)
  features.py      features partagées entraînement / API (anti training-serving skew)
  train.py         entraînement, baseline, artefact + metadata.json, option MLflow
  api/main.py      FastAPI : /health, /ready, /v1/model, /v1/predict, /v1/predict/batch
  api/schemas.py   contrat d'entrée / sortie (Pydantic)
tests/             pytest (features, API)
docs/              templates : model card, runbook, ADR
exercices/         démo pickle (S1)
```

## Usage de l'IA générative

Chaque activité indique son mode : **sans IA**, **IA déclarée** ou **IA imposée**.
En mode IA déclarée, ajoutez dans la description de vos commits ou de votre rendu :
outil utilisé, ce que vous lui avez demandé, ce que vous avez vérifié ou corrigé.

## Modèles analytiques dbt

Après l'installation des dépendances de développement, générez la zone bronze puis construisez
les modèles depuis la racine du dépôt :

```bash
pip install -r requirements-dev.txt
pip install -e .
python -m velov.data --bronze-out data/bronze.csv
dbt build --project-dir dbt --profiles-dir dbt
```

Le modèle staging lit `data/bronze.csv`, puis les modèles silver et gold exposent les observations
nettoyées. Le modèle gold applique le contrat défini par les six colonnes de `BRONZE_COLUMNS`
dans `src/velov/data.py`. Par défaut, dbt utilise une base DuckDB locale sous `dbt/target/`.
Pour utiliser un autre fichier bronze, passez `--vars '{"velov_bronze_csv": "chemin/vers/bronze.csv"}'`
à la commande `dbt build`.
