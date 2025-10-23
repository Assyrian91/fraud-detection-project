"""
Monitoring Dashboard for Fraud Detection System
"""
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path

# Initialize Dash app
app = dash.Dash(__name__)

# Load metrics
metrics_file = Path("reports/metrics/training_results.json")
if metrics_file.exists():
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
else:
    metrics_data = {}

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("🛡️ Fraud Detection Monitoring Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.Hr()
    ]),
    
    # Summary Cards
    html.Div([
        html.Div([
            html.H3("Model Accuracy"),
            html.H2(f"{metrics_data.get('Random Forest', {}).get('test_metrics', {}).get('accuracy', 0)*100:.2f}%",
                   style={'color': '#27ae60'})
        ], className='card'),
        
        html.Div([
            html.H3("Precision"),
            html.H2(f"{metrics_data.get('Random Forest', {}).get('test_metrics', {}).get('precision', 0)*100:.2f}%",
                   style={'color': '#3498db'})
        ], className='card'),
        
        html.Div([
            html.H3("Recall"),
            html.H2(f"{metrics_data.get('Random Forest', {}).get('test_metrics', {}).get('recall', 0)*100:.2f}%",
                   style={'color': '#e74c3c'})
        ], className='card'),
        
        html.Div([
            html.H3("F1 Score"),
            html.H2(f"{metrics_data.get('Random Forest', {}).get('test_metrics', {}).get('f1', 0)*100:.2f}%",
                   style={'color': '#9b59b6'})
        ], className='card'),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
    # Charts
    html.Div([
        html.Div([
            dcc.Graph(id='model-comparison')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='confusion-matrix')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
    
    html.Div([
        html.Div([
            dcc.Graph(id='metrics-over-time')
        ], style={'width': '100%'}),
    ]),
    
    # Auto-refresh
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # 30 seconds
        n_intervals=0
    )
])

# Callbacks
@app.callback(
    Output('model-comparison', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_model_comparison(n):
    models = []
    metrics = []
    
    for model_name, data in metrics_data.items():
        if 'test_metrics' in data:
            models.append(model_name)
            metrics.append(data['test_metrics'].get('f1', 0))
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=metrics, marker_color='#3498db')
    ])
    
    fig.update_layout(
        title='Model Comparison (F1 Score)',
        xaxis_title='Model',
        yaxis_title='F1 Score',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('confusion-matrix', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_confusion_matrix(n):
    # Get confusion matrix from best model
    best_model = 'Random Forest'
    if best_model in metrics_data:
        cm = metrics_data[best_model].get('confusion_matrix', [[0, 0], [0, 0]])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Normal', 'Predicted Fraud'],
            y=['Actual Normal', 'Actual Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            template='plotly_white'
        )
        
        return fig
    
    return go.Figure()

@app.callback(
    Output('metrics-over-time', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_metrics_timeline(n):
    # Simulated timeline data
    dates = pd.date_range(start='2025-01-01', periods=10, freq='W')
    accuracy = [0.95, 0.96, 0.97, 0.965, 0.97, 0.975, 0.98, 0.98, 0.985, 0.99]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers', name='Accuracy'))
    
    fig.update_layout(
        title='Model Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Accuracy',
        template='plotly_white'
    )
    
    return fig

# CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Fraud Detection Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #ecf0f1;
            }
            .card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                width: 20%;
            }
            .card h3 {
                color: #7f8c8d;
                margin: 0;
                font-size: 14px;
            }
            .card h2 {
                margin: 10px 0 0 0;
                font-size: 32px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
