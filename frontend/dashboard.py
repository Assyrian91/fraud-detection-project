import dash
import requests
from dash import dcc, html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)
server = app.server  # for deployment if needed

app.layout = html.Div(
    [
        html.H2("Customer Churn Prediction"),
        html.Label("Customer ID"),
        dcc.Input(id="customer_id", type="text"),
        html.Br(),
        html.Br(),
        html.Label("Tenure"),
        dcc.Input(id="tenure", type="number"),
        html.Br(),
        html.Br(),
        html.Label("Monthly Charges"),
        dcc.Input(id="monthly_charges", type="number"),
        html.Br(),
        html.Br(),
        html.Label("Contract Type"),
        dcc.Dropdown(
            id="contract_type",
            options=[
                {"label": "Month-to-month", "value": "Month-to-month"},
                {"label": "One year", "value": "One year"},
                {"label": "Two year", "value": "Two year"},
            ],
            value="Month-to-month",
        ),
        html.Br(),
        html.Button("Predict", id="predict_btn"),
        html.Br(),
        html.Br(),
        html.Div(id="result"),
    ]
)


@app.callback(
    Output("result", "children"),
    Input("predict_btn", "n_clicks"),
    State("customer_id", "value"),
    State("tenure", "value"),
    State("monthly_charges", "value"),
    State("contract_type", "value"),
)
def predict(n_clicks, customer_id, tenure, monthly_charges, contract_type):
    if n_clicks is None:
        return ""

    payload = {
        "customer_id": customer_id,
        "features": {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract_type,
        },
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()
        return f"Churn: {result['churn']} | Confidence: {result['confidence']}"
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)
