from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Data preparation
data = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC Score"],
    "LR Train": [0.91, 0.91, 1.00, 0.95, 0.62],
    "LR Test": [0.90, 0.90, 1.00, 0.95, 0.57],
    "LR SM Train": [0.95, 0.98, 0.92, 0.95, 0.95],
    "LR SM Test": [0.87, 0.96, 0.89, 0.93, 0.81],
    "MNB Train": [0.92, 0.96, 0.88, 0.91, 0.92],
    "MNB Test": [0.83, 0.95, 0.85, 0.90, 0.76],
    "XGB Train": [0.94, 0.94, 0.95, 0.94, 0.94],
    "XGB Test": [0.88, 0.95, 0.92, 0.93, 0.76],
    "XGB HP Train": [0.81, 0.82, 0.80, 0.81, 0.81],
    "XGB HP Test": [0.77, 0.95, 0.79, 0.86, 0.71],
    "RF Train": [1.00, 1.00, 1.00, 1.00, 1.00],
    "RF Test": [0.90, 0.93, 0.96, 0.95, 0.69],
    "RF HP Train": [0.82, 0.78, 0.89, 0.83, 0.82],
    "RF HP Test": [0.84, 0.94, 0.88, 0.91, 0.70],
}
metrics_df = pd.DataFrame(data)

# Separate into Train and Test DataFrames
train_data = metrics_df[["Metric"] + [col for col in metrics_df.columns if "Train" in col]]
test_data = metrics_df[["Metric"] + [col for col in metrics_df.columns if "Test" in col]]

# Transpose for heatmap
metrics_df_transposed = metrics_df.set_index("Metric").T
metrics_df_transposed.index.name = "Model"

# Initialize Dash app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Model Metrics Visualization", style={'textAlign': 'center'}),

    # Line chart for selected model's Train and Test
    dcc.Graph(id="line-chart"),
    
    # Dropdown for selecting model
    dcc.Dropdown(
        id="model-dropdown",
        options=[
            {"label": "Logistic Regression (LR)", "value": "LR"},
            {"label": "Logistic Regression SMOTE (LR SM)", "value": "LR SM"},
            {"label": "XGBoost (XGB)", "value": "XGB"},
            {"label": "XGBoost Hyperparameter (XGB HP)", "value": "XGB HP"},
            {"label": "Multinomial Naive Bayes (MNB)", "value": "MNB"},
            {"label": "Random Forest (RF)", "value": "RF"},
            {"label": "Random Forest Hyperparameter (RF HP)", "value": "RF HP"},
        ],
        value="LR",  # Default value (Logistic Regression)
        clearable=False,
    ),

    html.H1("Model Metrics Visualization", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Color Scale:"),
        dcc.Dropdown(
            id="color-scale-dropdown",
            options=[
                {"label": "Viridis", "value": "Viridis"},
                {"label": "Plasma", "value": "Plasma"},
                {"label": "Cividis", "value": "Cividis"},
                {"label": "RdBu", "value": "RdBu"},
                {"label": "Inferno", "value": "Inferno"}
            ],
            value="Viridis",  # Default color scale
            clearable=False,
        ),
    ], style={'margin-bottom': '20px'}),

    dcc.Graph(id="heatmap", style={'display': 'flex', 'flex-grow': 1})
])

# Callback for line chart
@app.callback(
    Output("line-chart", "figure"),
    Input("model-dropdown", "value")
)
def update_line_chart(selected_model):
    # Create the proper column names for Train and Test
    train_column = f"{selected_model} Train"
    test_column = f"{selected_model} Test"

    # Check if the columns exist in the data
    if train_column not in train_data.columns or test_column not in test_data.columns:
        raise ValueError(f"Columns {train_column} or {test_column} not found in the data")

    # Extracting the data for the selected model for Train and Test
    train_data_model = train_data[["Metric", train_column]].set_index("Metric")
    test_data_model = test_data[["Metric", test_column]].set_index("Metric")

    # Create line chart with both Train and Test metrics
    fig = go.Figure()

    # Train line
    fig.add_trace(go.Scatter(
        x=train_data_model.index,
        y=train_data_model[train_column],
        mode='lines+markers',
        name=f'{selected_model} Train',
        line=dict(color='blue')
    ))

    # Test line
    fig.add_trace(go.Scatter(
        x=test_data_model.index,
        y=test_data_model[test_column],
        mode='lines+markers',
        name=f'{selected_model} Test',
        line=dict(color='red')
    ))

    # Update layout for better presentation
    fig.update_layout(
        title=f"Metrics for {selected_model}",
        xaxis_title="Metrics",
        yaxis_title="Score",
        xaxis=dict(tickmode="linear"),
        template="plotly",
        height=500
    )

    return fig

@app.callback(
    Output("heatmap", "figure"),
    Input("color-scale-dropdown", "value")
)

def update_heatmap(color_scale):
    # Create heatmap with updated color scale
    fig = px.imshow(
        metrics_df_transposed.values,
        labels={"x": "Metrics", "y": "Models", "color": "Score"},
        x=metrics_df_transposed.columns,
        y=metrics_df_transposed.index,
        color_continuous_scale=color_scale,
        text_auto=".2f",
        height= 800,
        width= 1500,
    )
    fig.update_layout(
        title="Model Comparison Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Models",
        height=900,
        width=1000,
        font=dict(size=12),
        autosize=True
    )
    return fig

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)