from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pathlib import Path
import joblib
import pandas as pd
from html import escape
from urllib.parse import parse_qs

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
FEATURE_COLUMNS = ["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]
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
        document.querySelector("form").addEventListener("submit", () => {
            const button = document.querySelector('button[type="submit"]');
            button.disabled = true;
            button.textContent = "Predicting...";
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
    return render_page()


@app.get("/predict")
def prediction_form():
    return render_page()


@app.post("/predict")
async def predict(request: Request):
    global prediction_count
    form_data = parse_qs((await request.body()).decode("utf-8"), keep_blank_values=True)
    values = {
        field: form_data.get(field, [""])[0]
        for field in FEATURE_COLUMNS
    }
    try:
        pclass = int(values["pclass"])
        age = float(values["age"])
        sibsp = int(values["sibsp"])
        parch = int(values["parch"])
        fare = float(values["fare"])
        sex = values["sex"]
        embarked = values["embarked"]
    except (KeyError, ValueError, IndexError):
        return render_page("Enter valid values for every field.", status_code=400, values=values)

    if not 1 <= pclass <= 3:
        return render_page("Passenger class must be 1, 2, or 3.", status_code=400, values=values)
    if not 0.1667 <= age <= 80:
        return render_page("Age must be between 0.1667 and 80.", status_code=400, values=values)
    if not 0 <= sibsp <= 8:
        return render_page("Siblings/spouses must be between 0 and 8.", status_code=400, values=values)
    if not 0 <= parch <= 9:
        return render_page("Parents/children must be between 0 and 9.", status_code=400, values=values)
    if not 0 <= fare <= 512.3292:
        return render_page("Fare must be between 0 and 512.3292.", status_code=400, values=values)
    if sex not in {"male", "female"}:
        return render_page("Sex must be male or female.", status_code=400, values=values)
    if embarked not in {"S", "C", "Q"}:
        return render_page("Embarkation must be S, C, or Q.", status_code=400, values=values)

    # FEATURE_COLUMNS = FEATURE_COLUMNS + ['name', 'ticket', 'cabin', 'boat', 'home.dest', 'body']
    passenger = pd.DataFrame([{
        "pclass": pclass,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "sex": sex,
        "embarked": embarked,
        "name": "",
        "ticket": "",
        "cabin": "",
        "boat": "",
        "home.dest": "",
        "body": ""
    }])
    
    prediction = joblib.load(MODEL_PATH).predict(passenger)[0]
    prediction_count += 1
    is_survived = "Survived" if str(prediction).strip() == "1" else "Did Not Survive"
    result = (
        f"Prediction #{prediction_count} | Predicted class: {prediction} ({is_survived})."
    )
    return render_page(result, values=values)





