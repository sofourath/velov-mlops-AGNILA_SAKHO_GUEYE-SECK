# Runbook : API velov-availability

Ce runbook couvre l'API FastAPI, PostgreSQL, MLflow et le flow Prefect sur la VM Linux.
Les chemins ci-dessous supposent un clone sous `/opt/velov` et un environnement virtuel `.venv`.

## 1. Démarrer / arrêter

```bash
cd /opt/velov
docker compose -f compose.yaml -f compose.mlflow.yaml up -d --build
docker compose -f compose.yaml -f compose.mlflow.yaml ps
docker compose -f compose.yaml -f compose.mlflow.yaml down
```

L'API lit le bundle dans le dossier hôte `models/`, monté en lecture seule dans son conteneur.
Ne lancez pas `down -v` sauf si vous souhaitez aussi supprimer les données PostgreSQL et les artefacts MLflow.

## 2. Vérifier le service

| Vérification | Commande | Résultat attendu |
|---|---|---|
| Liveness | `curl --fail http://127.0.0.1:8000/health` | `{"status":"ok"}` |
| Readiness et modèle courant | `curl --fail http://127.0.0.1:8000/ready` | HTTP 200 et `model_version` |
| Serveur MLflow | `curl --fail http://127.0.0.1:5000/version` | Version MLflow |
| État Docker | `docker compose ps` | API et db `healthy` / `running` |

Logs utiles :

```bash
docker compose logs --tail=100 api
docker compose -f compose.yaml -f compose.mlflow.yaml logs --tail=100 mlflow
docker compose logs --tail=100 db
```

## 3. Planification Prefect

Installer les dépendances de développement et d'orchestration sur la VM, puis conserver les deux processus Prefect
(serveur et déploiement) sous un gestionnaire de services tel que systemd :

```bash
cd /opt/velov
. .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-orchestration.txt
export PREFECT_API_URL=http://127.0.0.1:4200/api
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
prefect server start --host 127.0.0.1
```

Dans un second service/terminal, avec les mêmes variables d'environnement :

```bash
cd /opt/velov
. .venv/bin/activate
export PREFECT_API_URL=http://127.0.0.1:4200/api
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m velov.flow
```

Le flow s'exécute chaque lundi à 03:00 UTC : ingestion synthétique de 90 jours, `dbt build`,
entraînement et enregistrement de l'alias `challenger`. Il ne promeut pas le modèle
automatiquement. Vérifier les métriques du run et lancer la promotion seulement si le test
porte sur la même fenêtre que le champion :

```bash
python -m velov.tracking promote
```

Après promotion, l'alias `previous-champion` conserve le modèle précédent.

## 4. Déployer / revenir en arrière

Sur la VM, avec `MLFLOW_TRACKING_URI` pointant sur le serveur MLflow :

```bash
cd /opt/velov
. .venv/bin/activate
python -m velov.deploy --alias champion --model-dir models
docker compose up -d --force-recreate api
curl --fail http://127.0.0.1:8000/ready
```

L'outil vérifie l'empreinte SHA-256 du bundle avant remplacement et garde une copie locale
datée. Si `/ready` échoue ou si les prédictions ne sont pas conformes, restaurer le précédent
champion du registre puis recréer le conteneur :

```bash
python -m velov.deploy --alias previous-champion --model-dir models
docker compose up -d --force-recreate api
curl --fail http://127.0.0.1:8000/ready
```

Le rollback par alias est disponible après une promotion qui a remplacé un champion existant.
Pour le tout premier champion, il n'existe pas encore d'alias `previous-champion`.

## 5. Incidents connus

| Symptôme | Cause probable | Diagnostic / action |
|---|---|---|
| `/ready` renvoie 503 | Bundle absent, empreinte incorrecte ou modèle incompatible | Vérifier `models/metadata.json`, l'artefact référencé et `docker compose logs api`; redéployer un alias connu |
| API indisponible après redémarrage | Conteneur/API ou PostgreSQL en erreur | `docker compose ps` puis `docker compose logs --tail=100 api db` |
| Déploiement d'alias échoue | Alias manquant, URI MLflow incorrecte ou bundle absent du run | Vérifier `MLFLOW_TRACKING_URI`, l'alias et les artefacts `package/` du run MLflow |
| Pas de nouveau challenger Prefect | Service Prefect arrêté ou tâche en échec | Vérifier les deux services Prefect, l'API à `127.0.0.1:4200/api` et les logs du flow |

## 6. Accès GitHub

Dans Settings → Branches, protéger `main`, exiger une pull request et les checks `lint`,
`tests`, `gitleaks` et `image`. Ces paramètres doivent être activés avec un compte ayant les
droits d'administration du dépôt ; le workflow ne peut pas les imposer lui-même.
