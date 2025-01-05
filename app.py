from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go

import dash

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

import numpy as np

# Load data
data = pd.read_csv('cleaned_collisions.csv')
data_cleaned_collisions_unencoded = pd.read_csv('cleaned_collisions_unencoded.csv')

collisions_data = pd.read_csv('collisions.csv')

collisions_data['Road Location'] = collisions_data['Road Location'].apply(
    lambda x: 'highway' if isinstance(x, str) and x.startswith('hwy ') else x
)
# Get the top 10 roads by count
final_grouped_data = collisions_data.groupby('Road Location').size().reset_index(name='Count')
final_sorted_data = final_grouped_data.sort_values(by='Count', ascending=False)
top_10_roads = final_sorted_data.head(10)['Road Location'].tolist()
top_10_data = collisions_data[collisions_data['Road Location'].isin(top_10_roads)].dropna(subset=['Latitude WGS84', 'Longitude WGS84'])


coords = data[['Latitude WGS84', 'Longitude WGS84']].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"],
            assets_folder="assets")
app.config.suppress_callback_exceptions = True

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Tabs(
                [
                    dbc.Tab(label="Collision Stats", tab_id="stats", tab_style={"color": "white"}),
                    dbc.Tab(label="Trends Visualization", tab_id="trends_visualization", tab_style={"color": "white"}),
                    dbc.Tab(label="EDA", tab_id="eda", tab_style={"color": "white"}),
                    dbc.Tab(label="Hot Routes", tab_id="hot_routes", tab_style={"color": "white"}),
                    dbc.Tab(label="Clustering", tab_id="clustering", tab_style={"color": "white"}),
                    dbc.Tab(label="Prediction", tab_id="prediction", tab_style={"color": "white"}),
                ],
                id="tabs",
                active_tab="stats",
                style={"background-color": "#2c3e50", "border-radius": "5px", "padding": "5px"}
            )
        ),
        html.Div(id="content", style={"padding": "20px"})
    ],
    fluid=True
)




# Callbacks to render the content of each tab
@app.callback(
    Output("content", "children"),
    [Input("tabs", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "eda":
        return create_eda_tab()
    elif active_tab == "hot_routes": 
        return create_hot_routes_tab()
    elif active_tab == "stats":
        return create_stats_tab()
    elif active_tab == "trends_visualization":
        return create_trends_visualization_tab()
    elif active_tab == "clustering":
        return create_clustering_tab()
    elif active_tab == "prediction":
        return create_prediction_tab()

    return html.Div("Select a tab", className="text-center")



# ----------------- Tab 1: Collision stats -----------------
def create_stats_tab():
    # Year Selector with enhanced design
    year_selector = dbc.Card(
        dbc.CardBody(
            [
                html.H5("Select Year", className="card-title text-center text-primary"),
                dcc.RadioItems(
                    id="year-selector-stats",
                    options=[
                        {"label": str(int(year)), "value": int(year)}
                        for year in sorted(data["Year"].unique())
                    ],
                    value=int(data["Year"].unique()[0]),
                    labelStyle={
                        "display": "inline-block",
                        "padding": "10px 15px",
                        "margin": "5px",
                        "fontSize": "16px",
                        "borderRadius": "20px",
                        "border": "1px solid #007bff",
                        "cursor": "pointer",
                        "backgroundColor": "#f8f9fa",
                        "transition": "0.3s",
                    },
                    inputStyle={"margin-right": "10px"},
                ),
            ]
        ),
        className="shadow-sm mb-4",
    )

    # Icons for the stats categories
    icons = {
        "fatal-injury": "fas fa-skull-crossbones",
        "non-fatal-injury": "fas fa-heartbeat",
        "pedestrian": "fas fa-walking",
        "bicycle": "fas fa-bicycle",
        "young-driver": "fas fa-child",
        "aggressive-driver": "fas fa-bolt",
        "distracted-driver": "fas fa-phone-alt",
        "impaired-driver": "fas fa-beer",
        "intersection": "fas fa-road",
        "total-collision": "fas fa-cogs",
    }

    def create_stat_card(title, icon_class, count_id):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.I(className=f"{icon_class} fs-3 text-primary mb-2 hover-grow"),
                                html.H6(title, className="card-title mb-2 text-center"),
                                html.P(id=count_id, className="card-text fs-4 text-center", style={"fontWeight": "bold", "color": "#007bff"}),
                            ],
                            style={"textAlign": "center", "display": "flex", "flexDirection": "column", "alignItems": "center"},
                        )
                    ]
                ),
                className="mb-4 shadow hover-card",
                style={"minHeight": "200px", "overflow": "hidden", "cursor": "pointer"},
            ),
            width=3,
        )

    # Define rows with an even distribution of columns
    stats_cards = dbc.Row(
        [
            create_stat_card("Fatal Injuries", icons["fatal-injury"], "fatal-injury-count"),
            create_stat_card("Non-Fatal Injuries", icons["non-fatal-injury"], "non-fatal-injury-count"),
            create_stat_card("Pedestrian Collisions", icons["pedestrian"], "pedestrian-count"),
            create_stat_card("Bicycle Collisions", icons["bicycle"], "bicycle-count"),
        ],
        className="mb-4 justify-content-center",
    )

    stats_cards_row_2 = dbc.Row(
        [
            create_stat_card("Young Drivers", icons["young-driver"], "young-driver-count"),
            create_stat_card("Aggressive Driving", icons["aggressive-driver"], "aggressive-driver-count"),
            create_stat_card("Distracted Driving", icons["distracted-driver"], "distracted-driver-count"),
            create_stat_card("Impaired Driving", icons["impaired-driver"], "impaired-driver-count"),
        ],
        className="mb-4 justify-content-center",
    )

    stats_cards_row_3 = dbc.Row(
        [
            create_stat_card("Intersection Collisions", icons["intersection"], "intersection-collision-count"),
            create_stat_card("Total Collisions", icons["total-collision"], "total-collision-count"),
        ],
        className="mb-4 justify-content-center",
    )

    # Wrap content within a fluid container to use the full width
    return dbc.Container(
        [
            html.H3("Collision Statistics", className="text-center text-primary my-4 fw-bold"),
            year_selector,
            stats_cards,
            stats_cards_row_2,
            stats_cards_row_3,
        ],
        fluid=True,
    )

@app.callback(
    [
        Output("fatal-injury-count", "children"),
        Output("non-fatal-injury-count", "children"),
        Output("pedestrian-count", "children"),
        Output("bicycle-count", "children"),
        Output("young-driver-count", "children"),
        Output("aggressive-driver-count", "children"),
        Output("distracted-driver-count", "children"),
        Output("impaired-driver-count", "children"),
        Output("intersection-collision-count", "children"),
        Output("total-collision-count", "children"),
    ],
    [Input("year-selector-stats", "value")],
)
def update_stats(year_selected):
    # Filter the data based on the selected year
    filtered_data = data[data["Year"] == year_selected]

    # Compute the statistics
    fatal_injuries = filtered_data["Fatal Injury"].sum()
    non_fatal_injuries = len(filtered_data) - fatal_injuries
    pedestrian_collisions = filtered_data["Pedestrian Collision"].sum()
    bicycle_collisions = filtered_data["Bicycle Collision"].sum()
    young_drivers = filtered_data["Young Demographic"].sum()
    aggressive_driving = filtered_data["Aggressive Driving"].sum()
    distracted_driving = filtered_data["Distracted Driving"].sum()
    impaired_driving = filtered_data["Impaired Driving"].sum()
    intersection_collisions = filtered_data["Intersection Collision"].sum()
    total_collisions = len(filtered_data)

    # Return the statistics to the cards
    return (
        f"{fatal_injuries}",
        f"{non_fatal_injuries}",
        f"{pedestrian_collisions}",
        f"{bicycle_collisions}",
        f"{young_drivers}",
        f"{aggressive_driving}",
        f"{distracted_driving}",
        f"{impaired_driving}",
        f"{intersection_collisions}",
        f"{total_collisions}",
    )



# -------Tab 2 - Trend Visualisation ---------------------
def create_trends_visualization_tab():
    return dbc.Container(
        [
            html.H3("Trends Visualization", className="text-center text-primary my-4 fw-bold"),
            
            # Filters Column
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Year", className="fw-bold"),
                            dcc.Dropdown(
                                id="filter-year",
                                options=[{'label': str(year), 'value': year} for year in sorted(data["Year"].unique())],
                                placeholder="Select Year",
                                multi=True,
                            ),
                            html.Label("Month", className="fw-bold mt-3"),
                            dcc.Dropdown(
                                id="filter-month",
                                options=[
                                    {'label': month, 'value': idx} for idx, month in enumerate(
                                        ["January", "February", "March", "April", "May", "June",
                                         "July", "August", "September", "October", "November", "December"], start=1
                                    )
                                ],
                                placeholder="Select Month",
                                multi=True,
                            ),
                            html.Label("Hour", className="fw-bold mt-3"),
                            dcc.RangeSlider(
                                id="filter-hour",
                                min=0, max=23,
                                step=1,
                                marks={i: f"{i}" for i in range(0, 24, 2)},
                                value=[0, 23],  # Default range: All hours
                            ),
                            html.Label("Feature Filters", className="fw-bold mt-3"),
                            dbc.Checklist(
                                options=[
                                    {"label": "Pedestrian Collision", "value": "Pedestrian Collision"},
                                    {"label": "Bicycle Collision", "value": "Bicycle Collision"},
                                    {"label": "Aggressive Driving", "value": "Aggressive Driving"},
                                    {"label": "Distracted Driving", "value": "Distracted Driving"},
                                    {"label": "Impaired Driving", "value": "Impaired Driving"},
                                    {"label": "Intersection Collision", "value": "Intersection Collision"},
                                ],
                                id="filter-features",
                                inline=False,
                            ),
                        ],
                        width=3,  # Filter column width
                        style={"borderRight": "1px solid #ccc", "padding": "10px"},
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    
dcc.Graph(

                                        id="heatmap-with-selection",
                                        style={"height": "600px"},
                                        config={"scrollZoom": True},  # Enable zooming for better exploration
                                    ),
                                    html.Div(
                                        "Select a region on the map to see trends.",
                                        className="text-secondary mt-3 text-center",
                                    ),
                                ]
                            ),
                            className="shadow-lg mb-4",
                        ),
                        width=9,  # Map column width
                    ),
                ],
                className="mb-4",
            ),

            # Line Chart and Stats Section
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    
dcc.Graph(

                                        id="line-chart-selected-data",
                                        style={"height": "400px"},
                                    ),
                                    html.Div(
                                        id="selected-region-stats",
                                        className="mt-4 text-primary",
                                        style={"fontSize": "16px"},
                                    ),
                                ]
                            ),
                            className="shadow-lg mb-4",
                        ),
                        width=12,
                    ),
                ],
            ),
            
            # Time/Day Trends Section
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Time/Day Trends", className="text-primary mt-5"),
                                    
dcc.Graph(

                                        id="time-day-chart",
                                        style={"height": "400px"},
                                    ),
                                    html.Div(
                                        id="highest-collision-details",
                                        className="mt-4",
                                    ),
                                ]
                            ),
                            className="shadow-lg mb-4",
                        ),
                        width=12,
                    ),
                ],
            ),
            
            # Additional Trends in Collision Causes
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    
dcc.Graph(

                                        id="line-chart-collision-causes",
                                        style={"height": "500px"},
                                    ),
                                ]
                            ),
                            className="shadow-lg mb-4",
                        ),
                        width=12,
                    ),
                ],
            ),
        ],
        fluid=True,
    )

@app.callback(
    [
        Output("heatmap-with-selection", "figure"),
        Output("line-chart-selected-data", "figure"),
        Output("selected-region-stats", "children"),
        Output("line-chart-collision-causes", "figure"),
        Output("time-day-chart", "figure"),  # Added
        Output("highest-collision-details", "children"),  # Added
    ],
    [
        Input("heatmap-with-selection", "selectedData"),
        Input("filter-year", "value"),
        Input("filter-month", "value"),
        Input("filter-hour", "value"),
        Input("filter-features", "value"),
    ]
)
def update_trends_visualization(selected_data, selected_years, selected_months, selected_hours, selected_features):
    # Start with the full dataset
    filtered_data = data.copy()

    # Apply Year, Month, Hour, and Feature filters
    if selected_years:
        filtered_data = filtered_data[filtered_data["Year"].isin(selected_years)]
    if selected_months:
        filtered_data = filtered_data[filtered_data["Month"].isin(selected_months)]
    if selected_hours:
        filtered_data = filtered_data[filtered_data["Hour"].between(selected_hours[0], selected_hours[1])]
    if selected_features:
        feature_conditions = [filtered_data[feature] == 1 for feature in selected_features]
        filtered_data = filtered_data[pd.concat(feature_conditions, axis=1).any(axis=1)]

    # Handle region selection from the heatmap
    if selected_data:
        lat_range = [point["lat"] for point in selected_data["points"]]
        lon_range = [point["lon"] for point in selected_data["points"]]
        filtered_data = filtered_data[
            (filtered_data["Latitude WGS84"].between(min(lat_range), max(lat_range))) &
            (filtered_data["Longitude WGS84"].between(min(lon_range), max(lon_range)))
        ]

    # Generate Scatter Map for Heatmap
    grouped_data = (
        filtered_data.groupby(['Latitude WGS84', 'Longitude WGS84'])
        .size()
        .reset_index(name='Collision Count')
    )
    scatter_map_fig = px.scatter_mapbox(
        grouped_data,
        lat="Latitude WGS84",
        lon="Longitude WGS84",
        size="Collision Count",
        size_max=20,
        color="Collision Count",
        color_continuous_scale=px.colors.sequential.Sunset,
        mapbox_style="carto-positron",
        zoom=10,
    )
    scatter_map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, dragmode="select")

    # Generate Line Chart for Collision Trends
    if not filtered_data.empty:
        trends = (
            filtered_data.groupby(["Year", "Month"])
            .size()
            .reset_index(name="Collision Count")
        )
        trends["Date"] = pd.to_datetime(trends[["Year", "Month"]].assign(day=1))
        line_chart_fig = px.line(
            trends,
            x="Date",
            y="Collision Count",
            title="Collision Trends for Selected Region",
            labels={"Collision Count": "Collisions"},
        )
    else:
        line_chart_fig = px.line(title="No Data Available for Selected Region")

    # Generate Statistics for Selected Region
    total_collisions = len(filtered_data)
    total_fatalities = filtered_data["Fatal Injury"].sum()
    yearly_counts = filtered_data.groupby("Year").size().reset_index(name="Count")
    yearly_stats = ", ".join(
        [f"{row['Year']}: {row['Count']} collisions" for _, row in yearly_counts.iterrows()]
    )
    stats_output = html.Ul(
        [
            html.Li(f"Total Collisions: {total_collisions}"),
            html.Li(f"Total Fatalities: {total_fatalities}"),
            html.Li(f"Yearly Breakdown: {yearly_stats}"),
        ]
    ) if not filtered_data.empty else "No collisions recorded in the selected region."

    # Generate Time/Day Trends
    day_labels = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}
    filtered_data["Day_of_Week"] = filtered_data["Day_of_Week"].map(day_labels)
    time_day_fig = px.bar(
        filtered_data.groupby("Hour").size().reset_index(name="Collision Count"),
        x="Hour",
        y="Collision Count",
        title="Collision Count by Hour of Day"
    )
    highest_day = filtered_data["Day_of_Week"].value_counts().idxmax() if not filtered_data.empty else "N/A"
    highest_day_count = filtered_data["Day_of_Week"].value_counts().max() if not filtered_data.empty else 0
    highest_hour = filtered_data["Hour"].value_counts().idxmax() if not filtered_data.empty else "N/A"
    highest_hour_count = filtered_data["Hour"].value_counts().max() if not filtered_data.empty else 0
    highest_collision_details = html.Div(
        [
            html.H5("Highest Collision Details"),
            html.P(f"Day with highest collisions: {highest_day} ({highest_day_count} collisions)"),
            html.P(f"Hour with highest collisions: {highest_hour}:00 ({highest_hour_count} collisions)"),
        ]
    )

    # Generate Collision Causes Chart
    collision_causes = filtered_data.groupby("Year")[
        ["Aggressive Driving", "Distracted Driving", "Impaired Driving", 
         "Pedestrian Collision", "Bicycle Collision", "Intersection Collision"]
    ].sum().reset_index()
    line_chart_causes = px.line(
        collision_causes,
        x="Year",
        y=["Aggressive Driving", "Distracted Driving", "Impaired Driving",
           "Pedestrian Collision", "Bicycle Collision", "Intersection Collision"],
        title="Trends in Primary Causes of Collisions Over Years",
        labels={"value": "Number of Incidents", "Year": "Year", "variable": "Cause Type"},
        markers=True
    )

    # Return all outputs
    return scatter_map_fig, line_chart_fig, stats_output, line_chart_causes, time_day_fig, highest_collision_details



# Tab 3: EDA
# ----------------- Tab 3: EDA -----------------
# ----------------- Tab 3: EDA -----------------
def create_eda_tab():
    # Dropdown for Condition Selection
    dropdown = dbc.Row(
        dbc.Col(
            dcc.Dropdown(
                id="eda-dropdown",
                options=[
                    {"label": "Weather Conditions", "value": "weather"},
                    {"label": "Light Conditions", "value": "light"},
                    {"label": "Artificial Light Conditions", "value": "artificial_light"},
                ],
                placeholder="Select a Condition to Explore",
                value="weather",
                style={"width": "50%"},
            ),
            width={"size": 8, "offset": 2},
        ),
        className="mb-4",
    )

    # Layout for EDA Tab
    return html.Div([
        html.H3("Feature Distribution and Collision Data", className="mb-4 text-primary text-center"),
        dropdown,
        dbc.Row([
            dbc.Col(dcc.Graph(id="eda-animated-graph", style={"height": "500px"}), width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="eda-correlation-heatmap", style={"height": "500px"}), width=12)
        ], className="mb-4"),  # Separate Row for Heatmap
        dbc.Row([
            dbc.Col(dcc.Graph(id="eda-top-correlations", style={"height": "500px"}), width=12)
        ])  # Separate Row for Top Correlations
    ])


# Callback for Animated Graph Based on Dropdown
@app.callback(
    Output("eda-animated-graph", "figure"),
    Input("eda-dropdown", "value"),
)
def update_animated_graph(selected_condition):
    try:
        if selected_condition == "weather":
            # Year-wise Weather Conditions Animation
            weather_columns = [
                'Weather Condition_Dust or smoke', 'Weather Condition_Fog, mist or smog',
                'Weather Condition_Rain', 'Weather Condition_Snow',
                'Weather Condition_Overcast or cloudy', 'Weather Condition_Freezing rain'
            ]
            weather_data = data.groupby('Year')[weather_columns].sum().reset_index()
            weather_data = weather_data.melt(id_vars='Year', var_name='Condition', value_name='Count')
            weather_data['Condition'] = weather_data['Condition'].str.replace('Weather Condition_', '')

            fig = px.bar(
                weather_data,
                x='Condition',
                y='Count',
                animation_frame='Year',
                title='Weather Conditions and Collision Density Over Years',
                labels={"Count": "Number of Collisions", "Condition": "Weather Condition"},
                template='plotly_dark',
                color='Count',
                color_continuous_scale=px.colors.sequential.Viridis
            )

        elif selected_condition == "light":
            # Year-wise Light Conditions Animation
            light_condition_data = data_cleaned_collisions_unencoded.groupby(['Year', 'Light Condition']).size().reset_index(name='Count')

            fig = px.bar(
                light_condition_data,
                x='Light Condition',
                y='Count',
                animation_frame='Year',
                title='Collision Density by Light Condition Over Years',
                labels={"Count": "Number of Collisions", "Light Condition": "Light Condition"},
                template='plotly_dark',
                color='Count',
                color_continuous_scale=px.colors.sequential.Plasma
            )

        elif selected_condition == "artificial_light":
            # Year-wise Artificial Light Conditions Animation
            artificial_light_condition_data = data_cleaned_collisions_unencoded.groupby(['Year', 'Artificial Light Condition']).size().reset_index(name='Count')

            fig = px.bar(
                artificial_light_condition_data,
                x='Artificial Light Condition',
                y='Count',
                animation_frame='Year',
                title='Collision Density by Artificial Light Condition Over Years',
                labels={"Count": "Number of Collisions", "Artificial Light Condition": "Artificial Light Condition"},
                template='plotly_dark',
                color='Count',
                color_continuous_scale=px.colors.sequential.Cividis
            )
        else:
            fig = go.Figure()

        fig.update_traces(marker=dict(line=dict(width=1.5, color='black')))
        fig.update_layout(
            title_font_size=20,
            title_font_color='cyan',
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font=dict(color='white'),
        )
        return fig

    except Exception as e:
        print(f"Error in update_animated_graph: {e}")
        return go.Figure().update_layout(
            title_text=f"Error: {str(e)}",
            title_font_color="red",
            paper_bgcolor="#1e2130",
            font=dict(color="white")
        )

# Callback for Static Visualizations (Correlations)
@app.callback(
    [Output("eda-correlation-heatmap", "figure"),
     Output("eda-top-correlations", "figure")],
    Input("eda-dropdown", "value")  # Added Input to ensure it updates
)
def update_static_visualizations(selected_condition):
    try:
        # Heatmap: Urban Settings vs. Collision Types
        collision_columns = [
            "Pedestrian Collision", "Aggressive Driving", "Distracted Driving",
            "Impaired Driving", "Bicycle Collision", "Intersection Collision"
        ]
        urban_setting_columns = [
            col for col in data.columns if col.startswith(
                ("Road Configuration", "Road Alignment", "Road Grade", "Road Surface", "Road Condition")
            )
        ]
        correlation_matrix = data[collision_columns + urban_setting_columns].corr().loc[urban_setting_columns, collision_columns]
        correlation_data = correlation_matrix.reset_index().melt(
            id_vars="index", var_name="Collision Type", value_name="Correlation"
        )
        correlation_data.rename(columns={"index": "Urban Setting"}, inplace=True)

        heatmap_fig = px.density_heatmap(
            correlation_data,
            x="Collision Type",
            y="Urban Setting",
            z="Correlation",
            title="Correlation Between Urban Settings and Collision Types",
            color_continuous_scale="Viridis",
            labels={"Urban Setting": "Urban Setting", "Collision Type": "Collision Type", "Correlation Coefficient": "Correlation"},
        )

        # Top Correlations
        top_correlations = correlation_data.sort_values(by="Correlation", ascending=False).head(10)
        top_corr_fig = px.bar(
            top_correlations,
            x="Correlation",
            y="Urban Setting",
            color="Collision Type",
            orientation="h",
            title="Top 10 Correlations Between Urban Settings and Collision Types",
            labels={"Urban Setting": "Urban Setting", "Collision Type": "Collision Type", "Correlation Coefficient": "Correlation"},
        )

        return heatmap_fig, top_corr_fig

    except Exception as e:
        print(f"Error in update_static_visualizations: {e}")
        return go.Figure(), go.Figure()


# Tab 4: Hot routes
def create_hot_routes_tab():
    # Create a bar chart figure for the top 10 roads
    fig_top_roads = px.bar(
        final_sorted_data.head(10),
        x='Road Location',
        y='Count',
        title="Top 10 Roads by Collision Count",
        color='Count',
        color_continuous_scale=px.colors.sequential.Sunset
    )
    fig_top_roads.update_layout(
        xaxis_title="Road Location",
        yaxis_title="Number of Collisions",
        hovermode="x unified"
    )

    return dbc.Container([
        html.H3("Interactive Top 10 Roads Collisions", className="text-center text-primary my-4"),

        html.Div(
            "Click on a bar below to highlight collisions on that road:",
            className="mb-3 text-center"
        ),

        
dcc.Graph(

            id='top-roads-bar-chart',
            figure=fig_top_roads,
            style={'height': '400px'}
        ),

        # Label for collision count
        html.Div(id='collision-count', style={'marginTop': '20px', 'fontSize': '18px', 'textAlign': 'center'}),

        # Map display
        
dcc.Graph(
id='collision-map', style={'height': '600px'})
    ], fluid=True)


# Update the callback to depend on clickData from the bar chart
@app.callback(
    [Output('collision-map', 'figure'),
     Output('collision-count', 'children')],
    [Input('top-roads-bar-chart', 'clickData')]
)
def update_map_and_count(clickData):
    if clickData and 'points' in clickData:
        # User has clicked on a bar
        selected_road = clickData['points'][0]['x']
        filtered_data = top_10_data[top_10_data['Road Location'] == selected_road]
    else:
        # No selection, show all top 10 roads
        filtered_data = top_10_data.copy()

    fig = px.scatter_mapbox(
        filtered_data,
        lat="Latitude WGS84",
        lon="Longitude WGS84",
        hover_name="Road Location",
        title="Collisions on Selected Road(s)",
        zoom=10,
        color="Road Location",
    )
    fig.update_layout(mapbox_style="carto-positron")

    collision_count = len(filtered_data)
    if clickData and 'points' in clickData:
        selected_road = clickData['points'][0]['x']
        count_label = f"Road: {selected_road}, Total Collisions: {collision_count}"
    else:
        count_label = f"All Top 10 Roads, Total Collisions: {collision_count}"

    return fig, count_label



# Tab 5: Clustering Analysis with Interactive Mapbox and Insights
def create_clustering_tab():
    return dbc.Container(
        [
            html.H3("Clustering Analysis", className="text-center text-primary my-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Select Year(s)", className="fw-bold"),
                            dcc.Dropdown(
                                id="year-filter",
                                options=[
                                    {'label': str(year), 'value': year} for year in sorted(data["Year"].unique())
                                ],
                                multi=True,
                                placeholder="Select Year(s)",
                                className="mb-3"
                            ),
                            html.Label("Epsilon (Radius of Cluster)", className="fw-bold"),
                            dcc.Slider(
                                id="eps-slider",
                                min=0.1, max=1.0, step=0.1, value=0.5,
                                marks={i / 10: str(i / 10) for i in range(1, 11)}
                            ),
                            html.Label("Min Samples (Cluster Density)", className="fw-bold mt-3"),
                            dcc.Slider(
                                id="min-samples-slider",
                                min=5, max=50, step=5, value=10,
                                marks={i: str(i) for i in range(5, 55, 5)}
                            ),
                            html.Button('Generate Clusters', id='cluster-button', className="mt-3 btn btn-primary"),
                        ],
                        width=3,
                        style={"borderRight": "1px solid #ccc", "padding": "10px"},
                    ),
                    dbc.Col(
                        dcc.Graph(id="mapbox-cluster-map", style={"height": "600px"}),
                        width=9,
                    )
                ],
            ),
            dbc.Row(
                dbc.Col(html.Div(id="cluster-summary", className="mt-4 text-center"), width=12),
            ),
            dbc.Row(
                dbc.Col(html.Div(id="cluster-insights", className="mt-4 text-primary"), width=12),
            )
        ],
        fluid=True,
    )

@app.callback(
    [
        Output("mapbox-cluster-map", "figure"),
        Output("cluster-summary", "children"),
        Output("cluster-insights", "children"),
    ],
    [Input("cluster-button", "n_clicks")],
    [
        State("year-filter", "value"),
        State("eps-slider", "value"),
        State("min-samples-slider", "value"),
    ]
)

def update_clustering_tab(n_clicks, selected_years, eps, min_samples):
    if n_clicks is None:
        return dash.no_update, "Adjust parameters and click Generate Clusters.", ""

    try:
        # Filter the dataset by selected years
        filtered_data = data.copy()
        if selected_years:
            filtered_data = filtered_data[filtered_data["Year"].isin(selected_years)]

        # Create a severity metric
        filtered_data['Severity_Metric'] = (
            filtered_data['Non Fatal Injury'] * 1 +
            filtered_data['Fatal Injury'] * 3 +
            filtered_data['Pedestrian Collision'] * 2 +
            filtered_data['Aggressive Driving'] * 2 +
            filtered_data['Impaired Driving'] * 1.5 +
            filtered_data['Distracted Driving'] * 1.5 +
            filtered_data['Bicycle Collision'] * 1 +
            filtered_data['Intersection Collision']* 2
        )

        # Select relevant columns for clustering
        clustering_data = filtered_data[['Latitude WGS84', 'Longitude WGS84', 'Severity_Metric']]

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        filtered_data['Cluster'] = dbscan.fit_predict(scaled_data)

        # Filter out noise clusters
        high_risk_clusters = filtered_data[filtered_data['Cluster'] != -1]

        # Generate interactive Mapbox visualization
        mapbox_fig = px.scatter_mapbox(
            filtered_data,
            lat="Latitude WGS84",
            lon="Longitude WGS84",
            color="Cluster",
            size="Severity_Metric",
            hover_data=["Severity_Metric", "Cluster"],
            title="Interactive Map of Clusters Based on Severity",
            mapbox_style="carto-positron",
            zoom=10,
        )

        # Summarize the clusters
        cluster_summary = high_risk_clusters['Cluster'].value_counts().reset_index()
        cluster_summary.columns = ["Cluster", "Count"]
        cluster_summary_text = f"Generated {len(cluster_summary)} clusters:\n{cluster_summary.to_dict()}"

        # Provide insights
        top_cluster = cluster_summary.iloc[0] if not cluster_summary.empty else None
        insights = []
        if top_cluster is not None:
            cluster_id = top_cluster["Cluster"]
            cluster_count = top_cluster["Count"]
            top_cluster_data = high_risk_clusters[high_risk_clusters["Cluster"] == cluster_id]
            avg_severity = top_cluster_data["Severity_Metric"].mean()
            insights.append(
                html.Div([
                    html.H4("Insights", className="text-center text-primary"),
                    html.P(f"Cluster {cluster_id} has the highest number of collisions ({cluster_count})."),
                    html.P(f"The average severity for this cluster is {avg_severity:.2f}."),
                    html.P(f"This cluster is centered around {top_cluster_data[['Latitude WGS84', 'Longitude WGS84']].mean().values}."),
                ])
            )
        else:
            insights.append(html.P("No significant clusters found."))

        return mapbox_fig, cluster_summary_text, insights

    except Exception as e:
        return dash.no_update, f"Error: {str(e)}", "An error occurred while generating clusters. Please try again."


# ----------------- Tab 6: Prediction -----------------
# Tab 6: Prediction
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

def create_prediction_tab():
    # Use the same dataset and logic for road options as Tab 4
    road_options = [
        {"label": road, "value": road}
        for road in collisions_data["Road Location"].dropna().unique()
    ]
    weather_options = [
        {"label": weather, "value": weather}
        for weather in collisions_data["Weather Condition"].dropna().unique()
    ]
    light_options = [
        {"label": light, "value": light}
        for light in collisions_data["Light Condition"].dropna().unique()
    ]

    # Layout
    return dbc.Container([
        html.H3("Likelihood of Collision", className="text-center mt-4"),
        dbc.Row([
            dbc.Col([
                html.Label("Select Road Name"),
                dcc.Dropdown(id="road-dropdown", options=road_options, placeholder="Select Road"),
                html.Label("Select Weather Condition"),
                dcc.Dropdown(id="weather-dropdown", options=weather_options, placeholder="Select Weather"),
                html.Label("Select Light Condition"),
                dcc.Dropdown(id="light-dropdown", options=light_options, placeholder="Select Light Condition"),
                dbc.Button("Calculate Likelihood", id="calculate-button", color="primary", className="mt-3"),
            ], width=4),
            dbc.Col([
                dcc.Graph(id="collision-likelihood-gauge", style={"height": "400px"})
            ], width=8),
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(
                    "Note: The likelihood is calculated based on historical data for the selected conditions.",
                    className="text-muted text-center mt-4",
                    style={"fontSize": "14px"}
                )
            )
        ])
    ], fluid=True)


@app.callback(
    Output("collision-likelihood-gauge", "figure"),
    [Input("calculate-button", "n_clicks")],
    [
        State("road-dropdown", "value"),
        State("weather-dropdown", "value"),
        State("light-dropdown", "value"),
    ]
)
def calculate_likelihood_with_smote(n_clicks, road, weather, light):
    if not n_clicks:
        # Default figure when no calculation is triggered
        return go.Figure().update_layout(
            title="Select inputs and click Calculate",
        )

    # Add 'Collision' column
    data = collisions_data.copy()
    data["Collision"] = 1  # All rows represent collisions

    # Create a diverse synthetic dataset for no-collision cases
    synthetic_no_collision = pd.DataFrame({
        "Road Location": np.random.choice(data["Road Location"].unique(), size=500, replace=True),
        "Weather Condition": np.random.choice(data["Weather Condition"].unique(), size=500, replace=True),
        "Light Condition": np.random.choice(data["Light Condition"].unique(), size=500, replace=True),
        "Collision": 0
    })
    data = pd.concat([data, synthetic_no_collision], ignore_index=True)

    # Encode features
    le_road = LabelEncoder()
    le_weather = LabelEncoder()
    le_light = LabelEncoder()

    data["Road Encoded"] = le_road.fit_transform(data["Road Location"].fillna("Unknown"))
    data["Weather Encoded"] = le_weather.fit_transform(data["Weather Condition"].fillna("Unknown"))
    data["Light Encoded"] = le_light.fit_transform(data["Light Condition"].fillna("Unknown"))

    # Prepare features and target
    X = data[["Road Encoded", "Weather Encoded", "Light Encoded"]]
    y = data["Collision"]

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X_balanced, y_balanced)

    # Prepare input data for prediction
    try:
        input_data = pd.DataFrame(
            {
                "Road Encoded": [le_road.transform([road])[0]] if road else [0],
                "Weather Encoded": [le_weather.transform([weather])[0]] if weather else [0],
                "Light Encoded": [le_light.transform([light])[0]] if light else [0],
            }
        )
    except ValueError as e:
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
        )

    # Predict likelihood
    predicted_proba = model.predict_proba(input_data)[0][1] * 100  # Probability of collision in percentage

    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_proba,
        title={"text": "Likelihood of Collision (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "red"},
            ],
        }
    ))

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8067)
