# ADR-0003 : Moteur d'inférence du modèle Vélo'v

- Statut : accepté
- Date : 2026-11-16
- Auteurs : équipe projet Vélo'v — TP5 — SAKHO_AGNILA_GUEYE-SECK

## Contexte
Les tests de charge réalisés en local montrent que, dans le protocole retenu, aucun des paliers testés à partir de 10 utilisateurs ne respecte le SLO de p95 inférieur ou égal à 200 ms.

La décomposition d'une prédiction avec `scripts/bench_predict.py` montre que, avec le moteur joblib, les étapes dominantes sont `model.predict` avec un p50 de 11,547 ms et `DataFrame + add_features` avec un p50 de 8,076 ms.

L'objectif est donc de réduire le coût de l'inférence du modèle sans dégrader significativement ses prédictions.

## Options envisagées
| Option | Avantages | Inconvénients |
|---|---|---|
| A. Rester sur scikit-learn (joblib) | Rien à changer, une seule pile | Latence d'inférence plus élevée sur les mesures locales |
| B. Exporter en ONNX, servir avec ONNX Runtime | Même modèle, artefact plus compact et `model.predict` environ 3,8 fois plus rapide en p50 sur une ligne | Une étape d'export et de vérification supplémentaire ; dépendances et versions à maîtriser |
| C. Plus de workers uvicorn | Aucune conversion | Ne change pas le coût par requête ; demande autant de cœurs |

## Triplet mesuré

Mesures locales sur la même machine, avec `scripts/bench_predict.py`, 500 mesures après 50 itérations d'échauffement.

| Mesure | joblib | ONNX | Commentaire |
|---|---:|---:|---|
| Écart maximal de prédiction | référence | 2,20 × 10⁻⁵ vélo | vérifié sur 43 200 lignes de l'historique simulé |
| Taille de l'artefact | 720 ko | 443 ko | ONNX est plus compact |
| `model.predict` p50, 1 ligne | 11,547 ms | 3,050 ms | ONNX ≈ 3,8× plus rapide |
| `model.predict` p95, 1 ligne | 16,327 ms | 5,851 ms | amélioration nette de la latence |
| Lot 100, temps par ligne | 0,1114 ms | 0,0391 ms | ONNX plus rapide |
| Lot 1000, temps par ligne | 0,0110 ms | 0,0116 ms | performances quasiment équivalentes |

La capacité sous le SLO n'a pas encore été remesurée avec ONNX avec les mêmes paliers Locust ; aucune valeur de capacité ONNX n'est donc revendiquée ici.

## Décision
Option B, avec `MODEL_FORMAT=onnx`. `velov.export_onnx` refuse l'export si une prédiction s'écarte de
plus de 0,001 vélo, sur toutes les lignes du jeu de données, et inscrit l'empreinte du fichier ONNX
dans `metadata.json` : l'API ne le charge qu'après vérification, comme le joblib.

## Conséquences

- L'export ONNX est vérifié avant utilisation : l'écart maximal mesuré sur les 43 200 lignes de l'historique simulé est de 2,20 × 10⁻⁵ vélo, très inférieur au seuil de refus fixé à 0,001 vélo.
- Avec ONNX, `model.predict` n'est plus l'étape dominante : son p50 passe de 11,547 ms à 3,050 ms, tandis que `DataFrame + add_features` mesure 9,569 ms en p50. La préparation des données devient donc le principal goulot mesuré dans le chemin de prédiction.
- L'artefact ONNX est plus compact : 443 ko contre 720 ko pour le modèle joblib.
- `protobuf` est épinglé à la version 6.33.5 dans `requirements-optim.txt` et `tests/test_onnx.py` vérifie l'export et la cohérence des prédictions.
- Les mesures présentées sont locales. La capacité sous le SLO avec ONNX doit encore être remesurée avec les mêmes paliers Locust avant de conclure sur la capacité de service.
