from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

import plotly.graph_objects as go

import dash

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

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

# Ensure Fatal Injury is binary (0 or 1)
data['Fatal Injury'] = data['Fatal Injury'].astype(int)

# Train Naive Bayes Model
features = [
    'Light Condition_Dawn', 'Light Condition_Daylight', 'Light Condition_Dusk',
    'Artificial Light Condition_Street lights on',
    'Weather Condition_Dust or smoke', 'Weather Condition_Fog, mist or smog', 
    'Weather Condition_Freezing rain', 'Weather Condition_Overcast or cloudy',
    'Weather Condition_Rain', 'Weather Condition_Snow', 'Weather Condition_Strong wind'
]
target = 'Fatal Injury'		  
																	

# Dynamically capture feature columns (all columns except target)
features = [col for col in data.columns if col != target]

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css", "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css"])
app.config.suppress_callback_exceptions = True

# Define the layout
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Tabs(
                [
                    dbc.Tab(label="Collision Stats", tab_id="stats"),
                    dbc.Tab(label="Trends Visualization", tab_id="trends_visualization"),
                    dbc.Tab(label="EDA", tab_id="eda"),
                    dbc.Tab(label="Hot Routes", tab_id="hot_routes"),   # Ensure this is present
                    dbc.Tab(label="Clustering", tab_id="clustering"),
                    dbc.Tab(label="Prediction", tab_id="prediction"),
                ],
                id="tabs",
                active_tab="stats",
            )
        ),
        html.Div(id="content")
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
    # RadioItems for year selection with enhanced design
    year_selector = html.Div(
        dcc.RadioItems(
            id='year-selector-stats',
            options=[{'label': str(int(year)), 'value': int(year)} for year in sorted(data['Year'].unique())],
            value=int(data['Year'].unique()[0]),  # Default to the first year
            labelStyle={
                'display': 'inline-flex',  # Inline-flex to keep alignment
                'alignItems': 'center',    # Center content
                'justifyContent': 'center',
                'padding': '10px 10px',    # Padding around each button
                'margin': '5px',           # Margin between buttons
                'fontSize': '16px',        # Font size
                'borderRadius': '20px',    # Rounded corners for buttons
                'border': '1px solid #007bff',  # Border color matching primary theme color
                'cursor': 'pointer',       # Pointer cursor for interactivity
                'backgroundColor': '#f8f9fa', # Light background for better aesthetics
            },
            inputStyle={
                "margin-right": "10px",  # Space between radio circle and text
            },
            className="mb-3"
        ),
        style={
            "textAlign": "center",     # Center align the whole component
            "marginTop": "5px",       # Space above the selector
        }
    )

    
    # Icons for the stats categories (Updated to Bootstrap Icons)
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
    
    # Card styling
    def create_stat_card(title, icon_class, count_id):
        return dbc.Col(
            dbc.Card(
                dbc.CardBody([ 
                    html.Div(
                        [
                            # Icon at the top
                            html.I(className=f"{icon_class} fs-3 text-primary"),  # Bootstrap icon with larger font size
                            # Smaller title
                            html.H6(title, className="card-title", style={"font-size": "0.9rem", "font-weight": "bold"}),
                            # Larger number
                            html.P(id=count_id, className="card-text", 
                                style={"font-size": "2rem", "font-weight": "bold", "color": "#333"}),
                        ],
                        style={"text-align": "center", "display": "flex", "flexDirection": "column", "alignItems": "center"}
                    ),
                ]),
                color="light", outline=True, className="mb-4 shadow-sm",
                style={"min-height": "200px", "overflow": "hidden"}  # Ensure card maintains height
            ), width=3  # Increase width to make full use of the row space
        )

    # Define rows with an even distribution of columns
    stats_cards = dbc.Row(
        [
            create_stat_card("Fatal Injuries", icons["fatal-injury"], "fatal-injury-count"),
            create_stat_card("Non-Fatal Injuries", icons["non-fatal-injury"], "non-fatal-injury-count"),
            create_stat_card("Pedestrian Collisions", icons["pedestrian"], "pedestrian-count"),
            create_stat_card("Bicycle Collisions", icons["bicycle"], "bicycle-count"),
        ],
        className="mb-4 justify-content-center"
    )

    stats_cards_row_2 = dbc.Row(
        [
            create_stat_card("Young Drivers", icons["young-driver"], "young-driver-count"),
            create_stat_card("Aggressive Driving", icons["aggressive-driver"], "aggressive-driver-count"),
            create_stat_card("Distracted Driving", icons["distracted-driver"], "distracted-driver-count"),
            create_stat_card("Impaired Driving", icons["impaired-driver"], "impaired-driver-count"),
        ],
        className="mb-4 justify-content-center"
    )
    
    stats_cards_row_3 = dbc.Row(
        [
            create_stat_card("Intersection Collisions", icons["intersection"], "intersection-collision-count"),
            create_stat_card("Total Collisions", icons["total-collision"], "total-collision-count"),
        ],
        className="mb-4 justify-content-center"
    )
    
    # Wrap content within a fluid container to use the full width
    return dbc.Container(
        [
            html.H3("Collision Statistics", className="text-center text-primary my-4"),
            year_selector,
            stats_cards,
            stats_cards_row_2,
            stats_cards_row_3
        ],
        fluid=True  # Make the container span the full width of the page
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
        Output("total-collision-count", "children")
    ],
    [Input("year-selector-stats", "value")]
)
def update_stats(year_selected):
    # Filter the data based on the selected year
    filtered_data = data[data['Year'] == year_selected]

    # Compute the statistics
    fatal_injuries = filtered_data['Fatal Injury'].sum()
    non_fatal_injuries = len(filtered_data) - fatal_injuries
    pedestrian_collisions = filtered_data['Pedestrian Collision'].sum()
    bicycle_collisions = filtered_data['Bicycle Collision'].sum()
    young_drivers = filtered_data['Young Demographic'].sum()
    aggressive_driving = filtered_data['Aggressive Driving'].sum()
    distracted_driving = filtered_data['Distracted Driving'].sum()
    impaired_driving = filtered_data['Impaired Driving'].sum()
    intersection_collisions = filtered_data['Intersection Collision'].sum()
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
        f"{total_collisions}"
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
                                marks={i: f"{i}:00" for i in range(0, 24, 2)},
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



# ----------------- Tab 3: EDA -----------------
def create_eda_tab():
    # Create visualizations for light and weather conditions using both datasets
    # Graph 1: Weather Conditions vs. Collision Density (data)
    weather_conditions = data[
        ['Weather Condition_Dust or smoke', 'Weather Condition_Fog, mist or smog',
         'Weather Condition_Rain', 'Weather Condition_Snow',
         'Weather Condition_Overcast or cloudy', 'Weather Condition_Freezing rain']
    ].sum()
    weather_conditions_df = weather_conditions.reset_index()
    weather_conditions_df.columns = ['Condition', 'Count']
    weather_conditions_df['Condition'] = weather_conditions_df['Condition'].str.replace('Weather Condition_', '')
    fig_weather = px.bar(
        weather_conditions_df,
        x='Condition',
        y='Count',
        title='Weather Conditions and Collision Density',
        labels={"Count": "Number of Collisions", "Condition": "Weather Condition"},
        template='plotly_dark',
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_weather.update_layout(
        title_font_size=20,
        title_font_color='cyan',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white')
    )

    # Graph 2: Collision Density by Light Condition (data_cleaned_collisions_unencoded)
    light_condition_counts = data_cleaned_collisions_unencoded['Light Condition'].value_counts().reset_index()
    light_condition_counts.columns = ['Light Condition', 'Count']
    fig_light_condition = px.bar(
        light_condition_counts,
        x='Light Condition',
        y='Count',
        title='Collision Density by Light Condition',
        labels={"Count": "Number of Collisions", "Natural Light Condition": "Light Condition"},
        template='plotly_dark',
        color='Count',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig_light_condition.update_layout(
        title_font_size=20,
        title_font_color='orange',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white')
    )

    # Graph 3: Collision Density by Artificial Light Condition (data_cleaned_collisions_unencoded)
    artificial_light_condition_counts = data_cleaned_collisions_unencoded['Artificial Light Condition'].value_counts().reset_index()
    artificial_light_condition_counts.columns = ['Artificial Light Condition', 'Count']
    fig_artificial_light_condition = px.bar(
        artificial_light_condition_counts,
        x='Artificial Light Condition',
        y='Count',
        title='Collision Density by Artificial Light Condition',
        labels={"Count": "Number of Collisions", "Artificial Light Condition": "Artificial Light Condition"},
        template='plotly_dark',
        color='Count',
        color_continuous_scale=px.colors.sequential.Cividis
    )
    fig_artificial_light_condition.update_layout(
        title_font_size=20,
        title_font_color='lightgreen',
        xaxis_title_font_size=16,
        yaxis_title_font_size=16,
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white')
    )

    # Graph 4: Urban Settings vs. Collision Types Correlation Heatmap
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

    fig_correlation_heatmap = px.density_heatmap(
        correlation_data,
        x="Collision Type",
        y="Urban Setting",
        z="Correlation",
        title="Correlation Between Urban Settings and Collision Types",
        color_continuous_scale="Viridis",
        labels={"Urban Setting": "Urban Setting", "Collision Type": "Collision Type", "Correlation": "Correlation Coefficient"},
    )

    # Graph 5: Top 10 Correlations
    top_correlations = correlation_data.sort_values(by="Correlation", ascending=False).head(10)
    fig_top_correlation_bar_chart = px.bar(
        top_correlations,
        x="Correlation",
        y="Urban Setting",
        color="Collision Type",
        orientation="h",
        title="Top 10 Correlations Between Urban Settings and Collision Types",
        labels={"Urban Setting": "Urban Setting", "Collision Type": "Collision Type", "Correlation": "Correlation Coefficient"},
    )

    # Set fixed height for graphs to prevent elongation
    graph_style = {"height": "500px"}  # Set a fixed height for all graphs

    # Return the feature visualization tab content
    return html.Div([
        html.H3("Feature Distribution and Collision Data", className="mb-4 text-primary text-center"),
        dbc.Row([dbc.Col(dcc.Graph(figure=fig_weather, style=graph_style, clear_on_unhover=True), width=12)], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_light_condition, style=graph_style, clear_on_unhover=True), width=6),
            dbc.Col(dcc.Graph(figure=fig_artificial_light_condition, style=graph_style, clear_on_unhover=True), width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_correlation_heatmap, style=graph_style, clear_on_unhover=True), width=12),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_top_correlation_bar_chart, style=graph_style, clear_on_unhover=True), width=12),
        ]),
    ])


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
        dcc.Graph(id='collision-map', style={'height': '600px'})
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



# Tab 5: Clustering Analysis with Heat Map and All Features
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
                                min=0.1, max=1.0, step=0.1, value=0.3,
                                marks={i / 10: str(i / 10) for i in range(1, 11)}
                            ),
                            html.Label("Min Samples (Cluster Density)", className="fw-bold mt-3"),
                            dcc.Slider(
                                id="min-samples-slider",
                                min=5, max=50, step=5, value=10,
                                marks={i: str(i) for i in range(5, 55, 5)}
                            ),
                            html.Button('Generate Heatmap', id='cluster-button', className="mt-3 btn btn-primary"),
                        ],
                        width=3,
                        style={"borderRight": "1px solid #ccc", "padding": "10px"},
                    ),
                    dbc.Col(
                        dcc.Graph(id="heatmap-cluster-map", style={"height": "600px"}),
                        width=9,
                    )
                ],
            ),
            dbc.Row(
                dbc.Col(html.Div(id="cluster-summary", className="mt-4 text-center"), width=12),
            )
        ],
        fluid=True,
    )

@app.callback(
    [Output("heatmap-cluster-map", "figure"), Output("cluster-summary", "children")],
    [Input("cluster-button", "n_clicks")],
    [
        State("year-filter", "value"),
        State("eps-slider", "value"),
        State("min-samples-slider", "value"),
    ]
)
def update_clustering_tab(n_clicks, selected_years, eps, min_samples):
    if n_clicks is None:
        return dash.no_update, "Adjust parameters and click Generate Heatmap."
    
    # Filter the dataset by selected years
    filtered_data = data.copy()
    if selected_years:
        filtered_data = filtered_data[filtered_data["Year"].isin(selected_years)]

    # Select all numeric features for clustering
    clustering_data = filtered_data.select_dtypes(include=[float, int])

    # Scale all numeric features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)

    # Add clusters to the filtered dataset
    filtered_data["Cluster"] = clusters

    # Filter high-risk clusters (exclude noise, i.e., Cluster = -1)
    # high_risk_clusters = filtered_data[filtered_data["Cluster"] != -1]

    # Generate Heat Map
    heatmap_fig = px.density_mapbox(
        filtered_data,
        lat="Latitude WGS84",
        lon="Longitude WGS84",
        z=None,  # Optional: Add a weight like collision count
        radius=15,
        center={"lat": data["Latitude WGS84"].mean(), "lon": data["Longitude WGS84"].mean()},
        zoom=10,
        mapbox_style="carto-positron",
        title="Heatmap of High-Risk Collision Areas"
    )

    # Create a cluster summary
    cluster_summary = filtered_data["Cluster"].value_counts().reset_index()
    cluster_summary.columns = ["Cluster", "Count"]
    cluster_summary_text = f"Generated {len(cluster_summary)} clusters:\n{cluster_summary.to_dict()}"

    return heatmap_fig, cluster_summary_text




# ---------------- Incorporate CSI and Prediction Logic from Snippet ----------------

# Define a Composite Severity Index (CSI) function
def calculate_csi(row):
    # Example weighting scheme - adjust as needed
    fatal = 4 if 'Fatal Injury' in row and row['Fatal Injury'] == 'Yes' else 0
    non_fatal = 2 if 'Non Fatal Injury' in row and row['Non Fatal Injury'] == 'Yes' else 0
    ped = 3 if 'Pedestrian Collision' in row and row['Pedestrian Collision'] == 'Yes' else 0
    bike = 3 if 'Bicycle Collision' in row and row['Bicycle Collision'] == 'Yes' else 0
    impaired = 2 if 'Impaired Driving' in row and row['Impaired Driving'] == 'Yes' else 0
    aggressive = 1 if 'Aggressive Driving' in row and row['Aggressive Driving'] == 'Yes' else 0
    distracted = 2 if 'Distracted Driving' in row and row['Distracted Driving'] == 'Yes' else 0
    intersection = 5 if 'Intersection Collision' in row and row['Intersection Collision'] == 'Yes' else 0

    return fatal + non_fatal + ped + bike + impaired + aggressive + distracted + intersection

# Compute CSI for each collision in the dataset
collisions_data['CSI'] = collisions_data.apply(calculate_csi, axis=1)

# Select relevant features and handle missing values
selected_features = ['Road Location', 'Road Condition', 'Weather Condition',
                     'Road Surface', 'Light Condition', 'CSI']
processed_data = collisions_data[selected_features].dropna()

# Encode categorical variables
encoded_data = pd.get_dummies(processed_data, columns=[
    'Road Condition', 'Weather Condition', 'Road Surface', 'Light Condition'
])

# Separate features and target for the initial model
X = encoded_data.drop(['Road Location', 'CSI'], axis=1)
y = encoded_data['CSI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MAE:", mae, "R2:", r2)

# Aggregate data by road location using CSI
road_group = collisions_data.groupby('Road Location').agg({
    'CSI': 'mean',  # Average CSI per road
    'Weather Condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'Road Surface': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'Light Condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
    'Road Condition': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
}).reset_index()

# Encode categorical variables for road-level modeling
encoded_road_data = pd.get_dummies(road_group, columns=[
    'Weather Condition', 'Road Surface', 'Light Condition', 'Road Condition'
])

# Separate features and target
X_road = encoded_road_data.drop(['Road Location', 'CSI'], axis=1)
y_road = encoded_road_data['CSI']

# Train-test split for road-level model
X_train_road, X_test_road, y_train_road, y_test_road = train_test_split(X_road, y_road, test_size=0.2, random_state=42)
road_model = RandomForestRegressor(n_estimators=100, random_state=42)
road_model.fit(X_train_road, y_train_road)

# Evaluate the model at the road level
y_pred_road = road_model.predict(X_test_road)
road_mae = mean_absolute_error(y_test_road, y_pred_road)
road_r2 = r2_score(y_test_road, y_pred_road)
print("Road-Level MAE:", road_mae, "Road-Level R2:", road_r2)

# Predict CSI for all roads
road_group['Predicted CSI'] = road_model.predict(X_road)
road_data_display = road_group[['Road Location', 'Predicted CSI']]


# ----------------- Tab 6: Prediction -----------------
def create_prediction_tab():
    # Create the layout from the snippet's logic
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("Road CSI Explorer", className="text-center mt-4"))),
        dbc.Row([
            dbc.Col([
                html.Label("Select Road:"),
                dcc.Dropdown(
                    id="road-selector",
                    options=[
                        {"label": road, "value": road}
                        for road in road_data_display['Road Location']
                    ],
                    placeholder="Select a road"
                ),
                dbc.Button("Get CSI", id="submit-button", color="primary", className="mt-3")
            ], width=4),
            dbc.Col([
                dcc.Graph(id="csi-gauge", style={"height": "400px"})
            ], width=8)
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="road-details", className="mt-4"))
        ])
    ], fluid=True)


@app.callback(
    [Output("csi-gauge", "figure"),
     Output("road-details", "children")],
    [Input("submit-button", "n_clicks")],
    [State("road-selector", "value")]
)
def update_road_csi(n_clicks, selected_road):
    if not selected_road:
        return go.Figure(), "Please select a road to view its CSI."

    # Get predicted CSI for the selected road
    selected_row = road_data_display[road_data_display['Road Location'] == selected_road]
    predicted_csi = selected_row['Predicted CSI'].values[0]*10

    # Determine a suitable range for the gauge
    max_csi = 20  
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_csi,
        title={"text": "Predicted CSI"},
        gauge={
            "axis": {"range": [0, max_csi]},
            "bar": {"color": "darkblue"},
        }
    ))

    details = f"The predicted CSI for '{selected_road}' is {predicted_csi:.2f}."
    return fig, details


#Tab 6: Road CSI


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8067)
