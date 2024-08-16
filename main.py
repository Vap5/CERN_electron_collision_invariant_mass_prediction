import streamlit as st
import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Define the LSTM model class 
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size)  # Additional layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer
    
    def forward(self, x):
        out, _ = self.lstm(x)
        if out.dim() == 3:
            out = self.relu(out[:, -1, :])  # Apply ReLU activation to the last time step
        elif out.dim() == 2:
            out = self.relu(out)  # Apply ReLU activation if no sequence dimension
        else:
            raise ValueError("Unexpected tensor dimensions from LSTM")
        out = self.fc1(out)  # First fully connected layer
        out = self.relu(out)  # ReLU activation for additional layer
        out = self.fc2(out)  # Output layer
        return out

# Hyperparameters


# Instantiate and train the model
model = EnhancedLSTMModel(input_size=16, hidden_size=128, num_layers=1) 
model.load_state_dict(torch.load(r'lstm_model.pth'))
model.eval()

# Function to predict with the model
def predict_invariant_mass(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        prediction = model(input_tensor).item()
    return prediction

# Streamlit UI
st.title("Invariant Mass Prediction (Electron Collision)")

st.write("Enter the features to predict the invariant mass. Please ensure the values are within the specified range.")

# Input fields
E1 = st.number_input("E1 (Energy of the first electron)", min_value=0.0, max_value=500.0, value=50.0)
px1 = st.number_input("px1 (Momentum component x of the first electron)", min_value=-100.0, max_value=100.0, value=0.0)
py1 = st.number_input("py1 (Momentum component y of the first electron)", min_value=-100.0, max_value=100.0, value=0.0)
pz1 = st.number_input("pz1 (Momentum component z of the first electron)", min_value=-500.0, max_value=500.0, value=0.0)
pt1 = st.number_input("pt1 (Transverse momentum of the first electron)", min_value=0.0, max_value=100.0, value=10.0)
eta1 = st.number_input("eta1 (Pseudorapidity of the first electron)", min_value=-5.0, max_value=5.0, value=0.0)
phi1 = st.number_input("phi1 (Azimuthal angle of the first electron)", min_value=-np.pi, max_value=np.pi, value=0.0)
Q1 = st.selectbox("Q1 (Charge of the first electron)", [-1, 1])

E2 = st.number_input("E2 (Energy of the second electron)", min_value=0.0, max_value=500.0, value=50.0)
px2 = st.number_input("px2 (Momentum component x of the second electron)", min_value=-100.0, max_value=100.0, value=0.0)
py2 = st.number_input("py2 (Momentum component y of the second electron)", min_value=-100.0, max_value=100.0, value=0.0)
pz2 = st.number_input("pz2 (Momentum component z of the second electron)", min_value=-500.0, max_value=500.0, value=0.0)
pt2 = st.number_input("pt2 (Transverse momentum of the second electron)", min_value=0.0, max_value=100.0, value=10.0)
eta2 = st.number_input("eta2 (Pseudorapidity of the second electron)", min_value=-5.0, max_value=5.0, value=0.0)
phi2 = st.number_input("phi2 (Azimuthal angle of the second electron)", min_value=-np.pi, max_value=np.pi, value=0.0)
Q2 = st.selectbox("Q2 (Charge of the second electron)", [-1, 1])

# Prepare the input data
input_data = [
    E1, px1, py1, pz1, pt1, eta1, phi1, Q1,
    E2, px2, py2, pz2, pt2, eta2, phi2, Q2
]

# Ensure categorical data (Q1, Q2) is properly encoded if needed
input_data[-2] = 0 if Q1 == -1 else 1  # Convert Q1 to 0 or 1
input_data[-1] = 0 if Q2 == -1 else 1  # Convert Q2 to 0 or 1

if st.button("Predict Invariant Mass"):
    prediction = predict_invariant_mass(model, input_data)
    st.write(f"Predicted Invariant Mass: {prediction:.2f} GeV")
    
    # Explanation of the result
    if prediction < 10:
        st.write("The predicted invariant mass is low, indicating a possible light resonance or background event.")
    elif 10 <= prediction < 50:
        st.write("The predicted invariant mass is moderate, which might indicate an intermediate resonance.")
    elif prediction >= 50:
        st.write("The predicted invariant mass is high, possibly indicating a heavy resonance like the Z boson.")
