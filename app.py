from fastapi import FastAPI, Query, File, UploadFile, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import time
import pymysql

from functions import NBAPredictor   # la classe O.O (Orientée Objet)



# Charger le prédicteur UNE seule fois
predictor = NBAPredictor()


# FastAPI initialization
app = FastAPI(
    title="NBA Prediction API",
    description="API de prédiction NBA by MP DATA",
    version="1.0.0"
)

# CORS autorisé pour tous
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Configuration de la connexion à la base de données ---
DB_HOST = "nba-project-db.cnukccqe83cq.eu-north-1.rds.amazonaws.com"  # Le point de terminaison de ton cluster
DB_USER = "admin"
DB_PASSWORD = "rootadmin"
DB_NAME = "nba_db"

def get_db_connection():
    return pymysql.connect(host=DB_HOST,
                           user=DB_USER,
                           password=DB_PASSWORD,
                           database=DB_NAME,
                           cursorclass=pymysql.cursors.DictCursor)


@app.middleware("http")
async def log_requests_and_responses(request: Request, call_next):
    start_time = time.time()
    
    # Lire le corps de la requête
    request_body = await request.body()
    
    response = await call_next(request)
    
    # Cloner le corps de la réponse pour le logging
    response_body_bytes = b""
    async for chunk in response.body_iterator:
        response_body_bytes += chunk
    
    process_time = (time.time() - start_time) * 1000

    def log_to_db():
        try:
            connection = get_db_connection()
            with connection.cursor() as cursor:
                sql = """
                INSERT INTO api_logs (request_path, request_method, request_payload, response_status_code, response_body, processing_time_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    str(request.url.path),
                    request.method,
                    request_body.decode('utf-8', errors='ignore'),
                    response.status_code,
                    response_body_bytes.decode('utf-8', errors='ignore'),
                    process_time
                ))
            connection.commit()
        except Exception as e:
            print(f"Erreur de logging dans la base de données: {e}")
        finally:
            if 'connection' in locals() and connection.open:
                connection.close()

    # Utiliser une BackgroundTask pour que le logging ne bloque pas la réponse
    return StreamingResponse(iter([response_body_bytes]), status_code=response.status_code, headers=dict(response.headers), background=BackgroundTask(log_to_db))


# Schéma d'entrée pour prédiction à partir de paramètres
class PlayerStats(BaseModel):
    GP: float
    MIN: float
    PTS: float
    FGM: float
    FGA: float
    FGP: float
    PM: float
    PA: float
    PAP: float
    FTM: float
    FTA: float
    FTP: float
    OREB: float
    DREB: float
    REB: float
    AST: float
    STL: float
    BLK: float
    TOV: float


# LES ROUTES API

@app.get("/")
def start_server():
    return {"message": "Le serveur NBA prediction conçu par Ketsia MULAPI a démarré !"}


@app.get("/api/nba/predict")
def predict_player(TOV: float = Query(...),
    GP: float = Query(...),
    MIN: float = Query(...),
    PTS: float = Query(...),
    FGM: float = Query(...),
    FGA: float = Query(...),
    FGP: float = Query(...),
    PM: float = Query(...),
    PA: float = Query(...),
    PAP: float = Query(...),
    FTM: float = Query(...),
    FTA: float = Query(...),
    FTP: float = Query(...),
    OREB: float = Query(...),
    DREB: float = Query(...),
    REB: float = Query(...),
    AST: float = Query(...),
    STL: float = Query(...),
    BLK: float = Query(...),
):

    arr = predictor.build_params(
        GP, MIN, PTS, FGM, FGA, FGP, PM, PA, PAP,
        FTM, FTA, FTP, OREB, DREB, REB, AST, STL, BLK, TOV
    )

    vect = NBAPredictor.preprocess(arr)
    pred = predictor.predict_vector(vect)

    return {"prediction": pred}


@app.get("/api/nba/info")
def decision_by_name(Name: str = Query(..., description="Nom du joueur")):
    """Prédiction à partir du nom dans le CSV."""
    return predictor.predict_by_name(Name)


@app.post("/api/nba/dataset")
def dataset_classification(file: UploadFile = File(...)):
    """Prédictions vectorisées sur un CSV uploadé."""

    # Vérification du format
    if not file.filename.endswith(".csv"):
        return {"error": "Le fichier doit être un CSV."}

    # Lecture du CSV
    df = pd.read_csv(file.file)

    # Prédictions (batch)
    return predictor.predict_dataset(df)