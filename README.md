# Titanic ML Pipeline and FastAPI Demo

A learning project exploring how to take a machine learning model beyond a notebook: build a scikit-learn pipeline, save the fitted model, load it from a separate application, and expose predictions through a FastAPI web form.

The prediction form collects passenger class, age, siblings/spouses aboard, parents/children aboard, fare, sex, and port of embarkation. The app validates these inputs, passes them to the saved model, and reports the predicted survival class. In this dataset, `0` means the passenger did not survive and `1` means the passenger survived.

## Learning Goals

- Build a preprocessing and model pipeline with scikit-learn.
- Train and evaluate a model in a Jupyter notebook.
- Persist a fitted model with `joblib` and load it in a separate Python application.
- Use FastAPI to serve predictions from a browser form.
- Validate user inputs and return prediction results to the user.

## Project Structure

```text
.
|-- pipelines.ipynb       # Model training and experimentation
|-- model.pkl             # Model artifact saved from the notebook
|-- sample1.csv           # Sample Titanic passenger data
`-- Deployment/
    |-- main.py            # FastAPI app and HTML prediction form
    |-- model.pkl          # Model artifact loaded by the app
    `-- sample1.csv        # Sample passenger data
```

The web app loads `Deployment/model.pkl`. If you retrain the notebook and want the app to use the new model, place the newly saved artifact at that path.

## Run the API

From the project root, create and activate a virtual environment, then install the app dependencies:

```powershell
cd Deployment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn pandas scikit-learn joblib
```

Start the development server:

```powershell
python -m uvicorn main:app --reload
```

Open <http://127.0.0.1:8000/> in a browser and submit the form. The form is served at `/` and `/predict`; predictions are submitted to `/predict` using HTTP POST. FastAPI's interactive API documentation is available at <http://127.0.0.1:8000/docs>.

The notebook downloads the Titanic dataset through scikit-learn's OpenML interface, so an internet connection may be required when running its data-loading cell.

## Model Artifact Compatibility

`joblib` artifacts should be loaded with compatible Python and scikit-learn versions to those used when the model was trained. If loading the artifact fails after changing environments, retrain and save it using the app's environment, or align the dependency versions between training and deployment.

## UI Attribution

The HTML form and its UI code in `Deployment/main.py` were generated with AI assistance as part of this learning project.
