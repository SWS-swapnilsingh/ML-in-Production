# Titanic Survival Prediction API

This project serves a saved scikit-learn model through FastAPI. It predicts whether a Titanic passenger survived (`1`) or did not survive (`0`) from passenger class, age, family aboard, fare, sex, and embarkation port.

## Example `POST /predict` JSON body

```json
{
    "pclass": 3,
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 32.2,
    "sex": "female",
    "embarked": "S"
}
```

Send it as JSON to `http://127.0.0.1:8000/predict` with `Content-Type: application/json`. The response includes the predicted class and its survival label. The browser form is available at `GET /predict`.

## Run locally (PowerShell)

```powershell
cd Deployment
python -m venv venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/predict> to use the form, or <http://127.0.0.1:8000/docs> to try the API. The UI code was generated with AI assistance.
