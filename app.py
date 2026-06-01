import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Breast Cancer ML Classifier", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('breast-cancer.csv')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

st.title("🧠 Breast Cancer Classification Web App")
st.markdown("Explore Support Vector Machines and other models on the Breast Cancer Wisconsin Diagnostic Dataset.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'breast-cancer.csv' exists in the app directory.")
    st.stop()

st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["SVM", "Random Forest", "Logistic Regression"])

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = None
if model_choice == "SVM":
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"])
    c_val = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
    model = SVC(kernel=kernel, C=c_val, random_state=42)
elif model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
else:
    c_val = st.sidebar.slider("Inverse Regularization (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=c_val, random_state=42, max_iter=1000)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{model_choice} Performance")
    st.metric("Accuracy", f"{acc:.4f}")
    
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

with col2:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    st.pyplot(fig)

st.subheader("2D PCA Decision Boundary Visualization")
st.markdown("We reduce the 30 features down to 2 Principal Components to visualize the model's decision boundary.")

# PCA for visualization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Re-train model on 2D data for boundary plotting
if model_choice == "SVM":
    model_2d = SVC(kernel=kernel, C=c_val, random_state=42)
elif model_choice == "Random Forest":
    model_2d = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
else:
    model_2d = LogisticRegression(C=c_val, random_state=42, max_iter=1000)

model_2d.fit(X_pca, y)

h = .02  
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
ax_pca.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
ax_pca.set_xlabel('Principal Component 1')
ax_pca.set_ylabel('Principal Component 2')
ax_pca.set_title(f'{model_choice} Decision Boundary')
handles, labels = scatter.legend_elements()
if len(handles) > 0:
    ax_pca.legend(handles, ['Benign', 'Malignant'][:len(handles)])
st.pyplot(fig_pca)
