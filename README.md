
#  PowerPulse: Household Energy Usage Forecast

This project focuses on analyzing and forecasting household energy consumption using real-world electricity usage data. By leveraging machine learning models and visualizations, it aims to encourage smarter energy practices, reduce costs, and support sustainability efforts.

---

##  Project Objective

- Analyze household energy usage patterns across time.
- Build and evaluate regression models to forecast future consumption.
- Provide actionable insights and visualizations for energy optimization.
- Assist stakeholders (homeowners, energy providers, policymakers) in informed decision-making.

---

##  Dataset Description

The dataset includes time-stamped household power consumption data with features such as:

- `Datetime`: Date and time of measurement.
- `Global_active_power`: Total power consumed (kilowatts).
- `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`: Energy consumption by specific appliances or zones.
- `Voltage`, `Global_intensity`, `Global_reactive_power`: Electrical attributes for deeper analysis.

---

##  ML Approach & Workflow

1. **Data Understanding & EDA** – Identifying trends, peak usage, anomalies.
2. **Preprocessing** – Handling missing values, feature creation, date-time parsing.
3. **Feature Engineering** – Rolling averages, peak hour flags, etc.
4. **Model Training** – Linear Regression, Random Forest, Neural Network.
5. **Model Evaluation** – RMSE, MAE, R² score used to compare performance.
6. **Dashboard & Visualization** – Final outputs and decision-support insights.

---

##  Best Performing Model

- **Model**: Neural Network 
- **Performance**:
  - RMSE: *Lowest among tested models*
  - R² Score: *Highest predictive accuracy*
  - Suitable for capturing non-linear patterns in energy usage.

---

## Key Insights

- Morning and evening consumption peaks identified.
- Weekend usage tends to be higher than weekdays.
- Appliance-level sub-metering helps in identifying energy-hungry devices.

---

##  Recommendations

- Encourage off-peak electricity usage to reduce bills.
- Adopt energy-efficient appliances in high-load zones.
- Use smart meters or home automation to monitor and optimize usage.

---

##  Business Applications

- **Homeowners**: Reduce energy bills, monitor usage.
- **Energy Providers**: Forecast demand, balance load, optimize tariffs.
- **Smart Grids**: Enable real-time analytics for better energy distribution.
- **Environment**: Lower carbon footprints through efficient energy practices.

---

##  Tech Stack

- **Languages**: Python, Pandas, NumPy
- **ML Models**: scikit-learn, Random Forest, Linear Regression, Neural Network
- **Visualization**: Matplotlib, Seaborn
- **Deployment (Optional)**: Streamlit Dashboard
- **Version Control**: Git, GitHub

---

##  Getting Started

```bash
# Clone the repo
git clone https://github.com/varshini-s-p/household-energy-forecast.git
cd household-energy-forecast

# Activate virtual environment
.\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run your main analysis or app
python model_evaluation.py
