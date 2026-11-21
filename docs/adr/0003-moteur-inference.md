# ADR-0003 : Moteur d'inférence du modèle Vélo'v

- Statut : accepté
- Date : 2026-11-16
- Auteurs : équipe pédagogique (exemple d'ADR rempli, EX-17)

## Contexte
Le test de charge de S5 montre que l'API tient le SLO (p95 sous 200 ms) jusqu'à environ 41 req/s sur
un cœur. La décomposition d'une prédiction (`scripts/bench_predict.py`) place le temps dans
`model.predict` (4,3 ms) et dans la construction du DataFrame et des features (3,4 ms). Le coût de
`model.predict` vient du pipeline scikit-learn (vérifications, ColumnTransformer, conversions),
conçu pour des lots : 4,3 ms pour 1 ligne, 5,0 ms pour 100.

## Options envisagées
| Option | Avantages | Inconvénients |
|---|---|---|
| A. Rester sur scikit-learn (joblib) | Rien à changer, une seule pile | Le plafond reste à environ 41 req/s par cœur |
| B. Exporter en ONNX, servir avec ONNX Runtime | Même modèle, prédiction 6 à 7 fois plus rapide dans l'API | Une étape d'export à vérifier ; outils d'export aux versions fragiles |
| C. Plus de workers uvicorn | Aucune conversion | Ne change pas le coût par requête ; demande autant de cœurs |

## Triplet mesuré (machine de préparation, API sur 1 cœur, Locust sur un autre)
| | joblib | ONNX | Comment |
|---|---|---|---|
| Écart de prédiction | référence | 2,2 × 10⁻⁵ vélo au maximum | 43 200 lignes de l'historique simulé |
| Taille | 720 ko | 443 ko | fichiers du paquet |
| `model.predict`, 1 ligne (p50) | 4,3 ms | 0,65 ms | `bench_predict.py --n 300` |
| Capacité sous le SLO | environ 41 req/s | environ 80 req/s | paliers de 30 s, 50 à 150 usagers |

## Décision
Option B, avec `MODEL_FORMAT=onnx`. `velov.export_onnx` refuse l'export si une prédiction s'écarte de
plus de 0,001 vélo, sur toutes les lignes du jeu de données, et inscrit l'empreinte du fichier ONNX
dans `metadata.json` : l'API ne le charge qu'après vérification, comme le joblib.

## Conséquences
- La vérification se fait à chaque export, pas une fois pour toutes. Sur un modèle entraîné sur
  30 jours, le même export donne un écart de 0,98 vélo sur 28 lignes : des températures tombent à
  moins de 10⁻⁶ d'un seuil, et en float32 l'arbre part de l'autre côté.
- Le gain est borné par le reste du chemin (loi d'Amdahl) : `add_features` (3,4 ms) devient l'étape
  la plus lente. C'est la prochaine optimisation, avec un test d'égalité contre la version pandas.
- protobuf est épinglé (6.33.5) dans `requirements-optim.txt` ; `tests/test_onnx.py` rejoue l'export en CI.
- Chiffres à remesurer sur la VM : ce sont ceux de la machine de préparation, avec des paliers courts.
