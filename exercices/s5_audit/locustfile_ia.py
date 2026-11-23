"""Test de charge généré par un assistant IA (exercice d'audit, S5).

Demande faite à l'assistant : « écris un test de charge Locust pour mon API FastAPI de prédiction
(POST /v1/predict, GET /health) et dis-moi si elle tient un SLO de 200 ms ».

Le script tourne sans erreur et affiche « SLO respecté ». À vous de dire pourquoi ce verdict ne vaut rien.

    locust -f exercices/s5_audit/locustfile_ia.py --headless -u 100 -r 100 -t 20s
"""

from datetime import UTC, datetime

from locust import HttpUser, constant, events, task

PAYLOAD = {
    "station_id": 1,
    "timestamp": "2026-11-17T08:00:00",
    "capacity": 20,
    "bikes_available": 10,
    "temperature": 12.0,
    "is_raining": False,
}


class ApiUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = constant(0)

    @task(9)
    def predict(self):
        payload = PAYLOAD.copy()
        payload["timestamp"] = datetime.now(UTC).isoformat()

        self.client.post("/v1/predict", json=payload)

    @task(1)
    def health(self):
        self.client.get("/health")


@events.test_stop.add_listener
def check_slo(environment, **kwargs):
    stats = environment.stats.entries[("/v1/predict", "POST")]
    p95 = stats.get_response_time_percentile(0.95)

    error_rate = stats.num_failures / stats.num_requests if stats.num_requests else 0

    print(f"p95 /v1/predict : {p95:.1f} ms")
    print(f"Taux d'erreur : {error_rate * 100:.2f} %")

    if p95 <= 200 and error_rate < 0.01:
        print("SLO respecté : l'API tient la charge")
    else:
        print("SLO non respecté")
