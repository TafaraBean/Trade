import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import threading

# Create a global Dash app instance
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='graph'),
    dcc.Interval(id='interval-component', interval=10*1000, n_intervals=0)  # Interval to trigger updates
])

# Create a placeholder for the figure
fig = go.Figure()

# Define the callback to update the figure
@app.callback(
    Output('graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_figure(n):
    return fig

# Function to update the global figure
def display_chart(new_fig):
    global fig
    fig = new_fig

# Function to run the Dash server
def run_server():
    app.run_server(debug=True, use_reloader=False)

# Start the Dash server in a separate thread
def start_visualization_server():
    threading.Thread(target=run_server).start()
