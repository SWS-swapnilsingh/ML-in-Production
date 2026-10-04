from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
import joblib
import pandas as pd
from html import escape
from pydantic import BaseModel, Field

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
FEATURE_COLUMNS = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]


class PassengerInput(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "pclass": 3,
                "age": 30,
                "sibsp": 0,
                "parch": 0,
                "fare": 32.2,
                "sex": "female",
                "embarked": "S",
            }]
        }
    }

    pclass: int = Field(ge=1, le=3)
    age: float = Field(ge=0.1667, le=80)
    sibsp: int = Field(ge=0, le=8)
    parch: int = Field(ge=0, le=9)
    fare: float = Field(ge=0, le=512.3292)
    sex: str = Field(pattern="^(male|female)$")
    embarked: str = Field(pattern="^(S|C|Q)$")


try:
    pipeline = joblib.load(MODEL_PATH)
except FileNotFoundError:
    pipeline = None

FORM_HTML = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Titanic Prediction</title>
    <style>
        body { max-width: 680px; margin: 40px auto; padding: 0 20px; font: 16px system-ui, sans-serif; color: #18212b; }
        h1 { margin-bottom: 8px; }
        form { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-top: 24px; }
        label { display: grid; gap: 6px; }
        input, select, button { box-sizing: border-box; min-height: 40px; padding: 8px; font: inherit; }
        input[type="range"] { min-height: 24px; padding: 0; accent-color: #147d64; }
        output { font-weight: 700; font-variant-numeric: tabular-nums; }
        button { grid-column: 1 / -1; cursor: pointer; }
        .message { padding: 12px; background: #eef4f1; border-left: 4px solid #147d64; }
        button:disabled { cursor: wait; opacity: 0.7; }
    </style>
</head>
<body>
    <h1>Titanic Prediction</h1>
    {message}
    <p class="message" id="result-message" role="status" aria-live="polite" hidden></p>
    <form method="post" action="/predict">
        <label>Passenger class <output id="pclass-value" for="pclass">3</output>
            <input id="pclass" name="pclass" type="range" min="1" max="3" step="1" value="__PCLASS__" required>
        </label>
        <label>Age <output id="age-value" for="age">30</output>
            <input id="age" name="age" type="range" min="0.1667" max="80" step="0.1" value="__AGE__" required>
        </label>
        <label>Siblings/spouses aboard <output id="sibsp-value" for="sibsp">0</output>
            <input id="sibsp" name="sibsp" type="range" min="0" max="8" step="1" value="__SIBSP__" required>
        </label>
        <label>Parents/children aboard <output id="parch-value" for="parch">0</output>
            <input id="parch" name="parch" type="range" min="0" max="9" step="1" value="__PARCH__" required>
        </label>
        <label>Fare <output id="fare-value" for="fare">32.20</output>
            <input id="fare" name="fare" type="range" min="0" max="512.3292" step="0.01" value="__FARE__" required>
        </label>
        <label>Sex
            <select name="sex" required>
                <option value="female" __FEMALE_SELECTED__>Female</option><option value="male" __MALE_SELECTED__>Male</option>
            </select>
        </label>
        <label>Port of embarkation
            <select name="embarked" required>
                <option value="S" __EMBARKED_S_SELECTED__>S</option><option value="C" __EMBARKED_C_SELECTED__>C</option><option value="Q" __EMBARKED_Q_SELECTED__>Q</option>
            </select>
        </label>
        <button type="submit">Predict</button>
    </form>
    <script>
        for (const field of document.querySelectorAll('input[type="range"]')) {
            const output = document.getElementById(`${field.id}-value`);
            const updateValue = () => {
                output.value = field.id === "fare" ? Number(field.value).toFixed(2) : field.value;
            };
            field.addEventListener("input", updateValue);
            updateValue();
        }
        document.querySelector("form").addEventListener("submit", async (event) => {
            event.preventDefault();
            const button = document.querySelector('button[type="submit"]');
            const message = document.getElementById("result-message");
            button.disabled = true;
            button.textContent = "Predicting...";
            message.hidden = false;
            message.textContent = "Predicting...";

            const values = Object.fromEntries(new FormData(event.currentTarget));
            for (const field of ["pclass", "age", "sibsp", "parch", "fare"]) {
                values[field] = Number(values[field]);
            }

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(values)
                });
                const result = await response.json();
                if (!response.ok) {
                    const detail = Array.isArray(result.detail)
                        ? result.detail.map(error => `${error.loc.at(-1)}: ${error.msg}`).join("; ")
                        : result.detail;
                    throw new Error(detail || "Prediction failed.");
                }
                message.textContent = `Prediction #${result.prediction_number} | Predicted class: ${result.prediction} (${result.survival}).`;
            } catch (error) {
                message.textContent = error.message || "Unable to get a prediction.";
            } finally {
                button.disabled = false;
                button.textContent = "Predict";
            }
        });
    </script>
</body>
</html>"""

prediction_count = 0


def render_page(message="", status_code=200, values=None):
    values = values or {
        "pclass": "3",
        "age": "30",
        "sibsp": "0",
        "parch": "0",
        "fare": "32.2",
        "sex": "female",
        "embarked": "S",
    }
    content = f'<p class="message">{escape(message)}</p>' if message else ""
    page = FORM_HTML.replace("{message}", content)
    replacements = {
        "__PCLASS__": escape(values["pclass"]),
        "__AGE__": escape(values["age"]),
        "__SIBSP__": escape(values["sibsp"]),
        "__PARCH__": escape(values["parch"]),
        "__FARE__": escape(values["fare"]),
        "__FEMALE_SELECTED__": "selected" if values["sex"] == "female" else "",
        "__MALE_SELECTED__": "selected" if values["sex"] == "male" else "",
        "__EMBARKED_S_SELECTED__": "selected" if values["embarked"] == "S" else "",
        "__EMBARKED_C_SELECTED__": "selected" if values["embarked"] == "C" else "",
        "__EMBARKED_Q_SELECTED__": "selected" if values["embarked"] == "Q" else "",
    }
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    return HTMLResponse(page, status_code=status_code)


@app.get("/")
async def root():
    return {"message": "Welcome to the Titanic Prediction API. Use /predict to make predictions. Use /docs for API documentation."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.get("/predict")
def prediction_form():
    return render_page()


@app.post("/predict")
def predict(passenger_input: PassengerInput):
    global prediction_count
    values = passenger_input.model_dump()
    # FEATURE_COLUMNS = FEATURE_COLUMNS + ['name', 'ticket', 'cabin', 'boat', 'home.dest', 'body']
    passenger = pd.DataFrame([{
        **values,
        "name": "",
        "ticket": "",
        "cabin": "",
        "boat": "",
        "home.dest": "",
        "body": ""
    }])
    
    prediction = str(pipeline.predict(passenger)[0]).strip()
    prediction_count += 1
    is_survived = "Survived" if prediction == "1" else "Did Not Survive"
    return {
        "prediction": prediction,
        "survival": is_survived,
        "prediction_number": prediction_count,
    }





