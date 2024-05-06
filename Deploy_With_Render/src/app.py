import pandas as pd  # Pandas for data manipulation
import dash  # Dash library for creating web applications
from dash import dcc, html, dash_table  # Components for building layout
from dash.dependencies import Input, Output  # Callbacks to update layout based on user input
import plotly.express as px  # Plotly Express for creating interactive visualizations
import plotly.graph_objects as go  # Plotly Graph Objects for more control over visualizations

 # Load the dataset
data_path = "Solar_Orbiter_With_Anomalies.csv"  # Path to dataset file
solar_data = pd.read_csv(data_path)  # Read dataset into DataFrame
   


"""
    Run the dashboard to visualize solar orbiter instrument data.

    It consists of several components:
  
    1. Instrument Checklist: Allows users to select instruments from a checklist.
    2. Date Range Picker: Enables users to select a range of dates for analysis.
    3. Time Series Chart: Displays the time series data for selected instruments.
    4. Correlation Heatmap: Shows the correlation between selected instruments.
    5. Anomaly Score Chart: Visualizes anomaly scores over time.
    6. SHAP Values: Provides Feature importance using SHAP Values.
    """


    # Initialize the Dash app
app = dash.Dash(__name__, title="Solar Orbiter Data Visualization") # Title of the Dash app which is showed in the browser tab
server = app.server
    
# Layout of the Dash app
app.layout = html.Div([
    html.H1("Solar Orbiter Instrument Data Visualization", style={'text-align': 'center'}),  # Title
    # Checklist to select instruments
    dcc.Checklist(
        id='instrument-checklist', # Component ID
        options=[{'label': col, 'value': col} for col in solar_data.columns[1:-2]],  # Options for checklist
        #I have removed the last two columns as they contain anomaly scores, which are not required for visualization.
        value=[solar_data.columns[1]],  # Default selected value (first instrument)
        inline=True
    ),
    # Date range picker
    dcc.DatePickerRange(
        id='date-picker-range',
        min_date_allowed=solar_data['Date'].min(),  # Minimum date allowed
        max_date_allowed=solar_data['Date'].max(),  # Maximum date allowed
        start_date=solar_data['Date'].min(),  # Default start date
        end_date=solar_data['Date'].max()  # Default end date
    ),
    # Two rows, each containing two graphs
    html.Div([
        html.Div([dcc.Graph(id='time-series-chart')], className="six columns"),  # Time Series Chart
        html.Div([dcc.Graph(id='correlation-heatmap')], className="six columns"),  # Correlation Heatmap
    ], className="row"),
    html.Div([
        html.Div([dcc.Graph(id='anomaly-score-chart')], className="six columns"),  # Anomaly Score Chart
    ], className="row"),
    html.Div(id='anomaly-stats', style={'margin-top': '20px', 'text-align': 'center'})  # Anomaly Stats
    ,
    html.Iframe(
        srcDoc=open("shap_values_plot.html").read(),
        style={"height": "500px", "width": "100%"}
    )
])
# Callbacks to update graphs
@app.callback(
    [Output('time-series-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('anomaly-score-chart', 'figure')],
    [Input('instrument-checklist', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)

def update_graphs(selected_instruments, start_date, end_date):
    """
    Callback function to update graphs based on user input.
    Args:
    selected_instruments (list): List of selected instruments.
    start_date (str): Start date selected by the user.
    end_date (str): End date selected by the user.
    Returns:
    figs (list): List of figures for each graph.
    """
    filtered_data = solar_data[(solar_data['Date'] >= start_date) & (solar_data['Date'] <= end_date)]  # Filtering data based on selected date range
    
    """This chart shows the time series data for the selected instruments."""
    """I chose this visualization because it is an effective way to show how the data changes over time."""
    time_series_fig = go.Figure()  # Creating a new figure for time series chart
    for instrument in selected_instruments:
        time_series_fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],  # X-axis data
                y=filtered_data[instrument],  # Y-axis data
                mode='lines+markers',  # Display mode
                name=instrument  # Instrument name
            )
        )
    time_series_fig.update_layout(title="Time Series of Selected Instruments")  # Updating layout of time series chart
    # Correlation Heatmap
    """The correlation heatmap is a graphical representation of the correlation matrix.
    I chose this visualization because it is an effective way to show the relationship between multiple variables."""
    correlation_fig = go.Figure(
        go.Heatmap(
            z=filtered_data[selected_instruments].corr(),  # Calculating correlation matrix
            x=selected_instruments,  # X-axis labels
            y=selected_instruments,  # Y-axis labels
            colorscale='Viridis'  # Color scale
        )
    )
    correlation_fig.update_layout(title="Correlation Heatmap")  # Updating layout of correlation heatmap
    """The anomaly score chart shows the anomaly scores over time."""
    """It shows how the anomaly scores change over time."""
    # Anomaly Score Chart

    anomaly_score_fig = go.Figure()  # Create a new figure for the anomaly score chart
    # Add traces for the anomaly score data
    # The trace defines how the data will be visualized, including style and condition-based formatting
    anomaly_score_fig.add_trace(go.Scatter(
    x=filtered_data['Date'],  # Set the x-axis as the Date column of the filtered data
    y=filtered_data['anomaly_score'],  # Set the y-axis as the anomaly_score column of the filtered data
    mode='lines+markers',  # Display both lines and markers on the graph
    name='Anomaly Score',  # Name the trace, which will appear in the legend
    marker=dict(
        color=[ 'red' if val < 0 else 'blue' for val in filtered_data['anomaly_score'] ],  # Use list comprehension to assign colors conditionally
        # Markers will be red if the anomaly score is below 0, otherwise blue
        size=5,  # Set the size of the markers
        line=dict(
            color='DarkSlateGrey',  # Color of the line around each marker
            width=2  # Width of the line around each marker
        )
    )
    ))

    # Update the layout of the figure to add titles and improve readability
    anomaly_score_fig.update_layout(
    title="Anomaly Scores Over Time (Lower the scores, higher chances of anomaly, negative score means definitely anomaly)",  # Main title of the chart
    xaxis_title='Date',  # Title for the x-axis
    yaxis_title='Anomaly Score'  # Title for the y-axis
    )
    return time_series_fig, correlation_fig, anomaly_score_fig  # Return updated figures
"""References:
1. https://dash.plotly.com/ - Dash Documentation
2. https://dash.plotly.com/layout - Dash Layout (HTML Components)
3. https://dash.plotly.com/dash-core-components - Dash Core Components ( DatePickerRange, Checklist)
4. https://dash.plotly.com/dash-html-components - Dash HTML Components (Div, H1 , Iframe)
5. https://plotly.com/python/plotly-express/ - Plotly Express ( px.line, px.scatter, px.bar)
6. https://plotly.com/python/graph-objects/ - Plotly Graph Objects ( go.Scatter, go.Heatmap, go.Figure)
7. https://www.coursera.org/projects/interactive-dashboards-plotly-dash?tab=guided-projects - Coursera Project

"""


if __name__ == "__main__":
   
    app.run_server(debug=True)  # Start the Dash server in debug mode
