# Runbook : API velov-availability

> Template à compléter au fil des sessions. Public cible : une personne d'astreinte
> qui ne connaît pas le projet et doit rétablir le service à 3 h du matin.

## 1. Vue d'ensemble
- Ce que fait le service, en deux phrases :
- Schéma d'architecture (lien) :
- Dépendances (base de données, stockage du modèle...) :

## 2. Démarrer / arrêter
```bash
# démarrer
# arrêter
# redémarrer un seul service
```

## 3. Vérifier que tout va bien
| Vérification | Commande | Résultat attendu |
|---|---|---|
| Le process répond | `curl -s http://<hôte>:8000/health` | `{"status":"ok"}` |
| Le modèle est chargé | `curl -s http://<hôte>:8000/ready` | HTTP 200 + version |
| | | |

## 4. Incidents connus
| Symptôme | Cause probable | Diagnostic | Action |
|---|---|---|---|
| `/ready` renvoie 503 | | | |
| Le conteneur redémarre en boucle | | | |
| Latence élevée | Saturation CPU / trop de requêtes concurrentes | Vérifier le p95 Locust, `X-Process-Time-Ms` et `docker stats` | Réduire la charge, puis tester la configuration à 2 workers / 2 CPU ; si le SLO reste dépassé, revenir à une charge inférieure |

### SLO et capacité mesurée — EX-16

SLO : au moins 95 % des requêtes `POST /v1/predict` doivent répondre en moins de 200 ms, avec un taux d'erreur inférieur à 1 %.

Les tests de charge ont été réalisés localement avec Locust.

Avec 1 worker / 1 CPU, aucun des paliers testés à 10, 25, 50, 100 et 150 utilisateurs ne respecte le SLO.

Avec 2 workers / 2 CPU, aucun de ces mêmes paliers ne respecte non plus le SLO.

La capacité sous SLO n'a donc pas été atteinte dans la plage testée : elle est inférieure au premier palier de 10 utilisateurs dans ce protocole.

Près du plafond :
- surveiller le p95 et le taux d'erreur ;
- vérifier la saturation CPU avec `docker stats` ;
- comparer `X-Process-Time-Ms` avec la latence Locust ;
- si le CPU est saturé, tester une augmentation contrôlée du nombre de workers et de CPU ;
- remesurer avec exactement le même protocole avant de conserver la modification.

## 5. Déployer une nouvelle version du modèle — EX-18

### Stratégie de remplacement

Une nouvelle version du modèle ne doit pas remplacer immédiatement la version actuellement servie.

Avant promotion :

1. vérifier que les tests automatisés passent ;
2. vérifier les métriques du nouveau modèle ;
3. comparer ses prédictions avec celles de la version de référence ;
4. mesurer sa latence avec le même protocole de benchmark ;
5. vérifier que le SLO de l'API reste respecté.

### Signaux et seuils de retour arrière

Déclencher un retour arrière si l'un des critères suivants est observé après déploiement :

- `/ready` renvoie HTTP 503 ;
- le taux d'erreur de `/v1/predict` atteint ou dépasse 1 % ;
- le p95 de `/v1/predict` dépasse 200 ms ;
- le nouveau modèle présente une régression de qualité par rapport au modèle validé ;
- le service devient instable ou le conteneur redémarre de manière répétée.

### Retour arrière (rollback)

Revenir à la dernière version du modèle précédemment validée, puis redémarrer l'API et vérifier :

```bash
curl -s http://<hôte>:8000/health
curl -s http://<hôte>:8000/ready
```
Après le rollback, relancer un test de prédiction et vérifier que le service respecte de nouveau le SLO avant de considérer l'incident comme résolu.

## 6. Contacts et escalade
-
