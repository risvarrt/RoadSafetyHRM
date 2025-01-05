# Road Safe HRM: Enhancing Road Safety Through Data-Driven Insights

## Overview

Road Safe HRM is an interactive tool designed to improve road safety in Halifax Regional Municipality (HRM) by analyzing historical traffic collision data. By leveraging advanced machine learning and visualization techniques, the project identifies high-risk areas, analyzes collision trends, and predicts potential risks under specific conditions.

---

## Features

- **Interactive Visualizations**:
  - Heatmaps and clustering to highlight accident hotspots.
  - Temporal analysis of collision trends by hour, day, and year.
  - Correlation heatmaps to uncover contributing factors to collisions.
- **Machine Learning Models**:
  - **DBSCAN Clustering**: Identifies accident-prone areas based on geospatial data.
  - **Naïve Bayes Prediction**: Calculates the likelihood of collisions under specific conditions.
- **Geospatial and Temporal Insights**:
  - Analysis of environmental factors like weather and lighting.
  - Identification of high-risk roads and intersections.
- **Customizability**:
  - Adjustable clustering parameters for fine-grained analysis.
  - Filters for targeted exploration of data subsets.

---

## Tech Stack

- **Programming Language**: Python
- **Visualization**:
  - [Plotly](https://plotly.com/python/)
  - [Dash](https://dash.plotly.com/)
  - Dash Bootstrap Components
- **Machine Learning**:
  - [Scikit-learn](https://scikit-learn.org/stable/)
  - SMOTE for class imbalance handling
- **Data Sources**:
  - [Halifax Open Data Portal](https://data-hrm.hub.arcgis.com/datasets/HRM::traffic-collisions/about)

---

## Architecture Diagram

*(Attach your architecture diagram here.)*

---

## How It Works

1. **Data Exploration**: Users interact with collision data via dynamic visualizations.
2. **Hotspot Identification**: DBSCAN clustering identifies areas with high collision severity.
3. **Predictive Analytics**: Naïve Bayes calculates collision likelihood, displayed via intuitive gauge charts.
4. **Interactive Modules**: Explore correlations, trends, and specific conditions influencing road safety.

---

## Key Modules

- **Collision Statistics**: Summarizes total collisions, fatalities, and trends by categories.
- **EDA (Exploratory Data Analysis)**: Animated graphs and correlation heatmaps to explore data distributions.
- **Hot Routes**: Highlights the most collision-prone roads and their specific risk factors.
- **Clustering Tab**: Uses DBSCAN to reveal accident clusters and their severity.
- **Collision Likelihood**: Predicts risk levels based on conditions like weather, lighting, and location.

---

## Dataset

- **Source**: Halifax Open Data Portal
- **Size**: 35,414 rows and 31 columns after preprocessing.
- **Key Features**:
  - Geospatial data (latitude, longitude)
  - Environmental factors (weather, lighting)
  - Temporal data (year, month, hour)
  - Collision severity metrics

---

## Future Enhancements

- Real-time data integration for dynamic monitoring.
- Advanced predictive models like Gradient Boosting or Neural Networks.
- IoT-enabled safety interventions via smart infrastructure.
- Inclusion of demographic and vehicle data for deeper insights.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contributing

We welcome contributions to improve Road Safe HRM! To get started:

1. Fork the repository from the [Git Repository](https://git.cs.dal.ca/himanshi/csci6612_groupproject).
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m "Add your feature or fix"`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

### Guidelines:
- Ensure your code adheres to the project's coding standards.
- Write clear and concise commit messages.
- Document any new features or significant changes in the README file.

---

## Contributors

- Himanshi Verma ([hm771904@dal.ca](mailto:hm771904@dal.ca))
- Ashish Kumar Guntipalli ([as589490@dal.ca](mailto:as589490@dal.ca))
- Rishi Varman Rathimala Thangaravi ([rs329673@dal.ca](mailto:rs329673@dal.ca))

