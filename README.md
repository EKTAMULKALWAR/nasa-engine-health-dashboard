# 🚀 NASA Engine Health Monitoring Dashboard

**Predictive maintenance system for turbofan engines using LSTM and NASA C-MAPSS data**

This project implements a complete **machine health monitoring system** that predicts the Remaining Useful Life (RUL) of jet engines using time-series sensor data and exposes the results through an interactive web dashboard with an AI assistant.

Built on NASA’s **C-MAPSS FD001** dataset, the system simulates how industrial AI is used in aerospace, healthcare devices, and manufacturing to detect failures before they happen.

---

## 🔍 What this system does

- Uses an **LSTM neural network** to predict Remaining Useful Life (RUL)  
- Applies **RUL capping** to focus on near-failure behavior  
- Classifies engines into **HIGH / MEDIUM / LOW risk**  
- Visualizes fleet health using charts and tables  
- Provides an **AI assistant** to query predictions in natural language  

This turns raw telemetry into **actionable maintenance decisions**.

---

## 🧠 Why this matters

In real-world systems (aircraft engines, medical devices, factory equipment), failure is expensive and dangerous.  
Predictive maintenance allows organizations to:

- Prevent breakdowns  
- Reduce downtime  
- Improve safety  
- Lower operational costs  

This project demonstrates how **machine learning + time-series modeling + analytics dashboards** are used to achieve that.

---

## 📊 Dataset

NASA C-MAPSS FD001  
Simulated turbofan engine degradation data with:

- 24 sensor readings  
- Multiple engines  
- Full life-to-failure trajectories  

Each engine’s **true RUL** is computed from its known failure cycle.

---

## ⚙️ Model

**LSTM (Long Short-Term Memory)** neural network trained on rolling windows of sensor data.

Input:
- Last 50 time steps  
- 24 sensor features  

Output:
- Predicted Remaining Useful Life (RUL)

RUL is capped to stabilize training and focus on degradation near failure.

---

## 📈 Performance

**RMSE (capped RUL): ~3.8 cycles**

This means the model predicts how long an engine will last within a few cycles of accuracy.

---

## 🖥️ Live Dashboard

The Streamlit app provides:

- Fleet-level health overview  
- Risk filtering (HIGH / MEDIUM / LOW)  
- Scatter plots of predicted vs true RUL  
- AI assistant for querying engine health  

Example queries:
- “How many engines are high risk?”  
- “Show the worst 10 engines.”  
- “What is the average RUL?”

---

## 🧱 Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras (LSTM)  
- Matplotlib  
- Streamlit  

---

## 🚀 Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
