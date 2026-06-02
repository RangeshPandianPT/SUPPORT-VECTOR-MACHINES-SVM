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
import joblib
import shap

st.set_page_config(page_title="Breast Cancer ML Classifier", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('breast-cancer.csv')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

st.title("🧠 Breast Cancer Classification Web App")
st.markdown("Explore Support Vector Machines, run live predictions, and analyze feature importance.")

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'breast-cancer.csv' exists.")
    st.stop()

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Benchmarking", "🔮 Live Prediction", "📈 Explainability (SHAP)", "📁 Custom Data Upload"])

with tab1:
    st.sidebar.header("Benchmarking Configuration")
    model_choice = st.sidebar.selectbox("Select Model for Benchmarking", ["SVM", "Random Forest", "Logistic Regression"])
    
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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

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

with tab2:
    st.header("Live Tumor Prediction")
    st.markdown("Use the optimized SVM model to predict diagnosis based on custom features.")
    
    try:
        best_model = joblib.load('best_svm_model.pkl')
        st.success("Loaded optimized SVM model successfully.")
        
        # Create input sliders for the top 5 most important features (for simplicity in UI)
        # We will set the rest to their mean values
        st.subheader("Input Top Features")
        col1, col2 = st.columns(2)
        
        input_data = {}
        for i, col_name in enumerate(X.columns):
            mean_val = float(X[col_name].mean())
            min_val = float(X[col_name].min())
            max_val = float(X[col_name].max())
            
            # Show sliders only for a few key features to not overwhelm the UI
            if i < 6:
                if i % 2 == 0:
                    with col1:
                        input_data[col_name] = st.slider(col_name, min_val, max_val, mean_val)
                else:
                    with col2:
                        input_data[col_name] = st.slider(col_name, min_val, max_val, mean_val)
            else:
                # Set others to mean silently
                input_data[col_name] = mean_val
                
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            pred = best_model.predict(input_df)[0]
            prob = best_model.predict_proba(input_df)[0]
            
            if pred == 1:
                st.error(f"Prediction: **Malignant** (Confidence: {prob[1]*100:.2f}%)")
            else:
                st.success(f"Prediction: **Benign** (Confidence: {prob[0]*100:.2f}%)")
                
    except Exception as e:
        st.warning(f"Could not load 'best_svm_model.pkl'. Did you run the training script? Error: {e}")

with tab3:
    st.header("Model Explainability (SHAP & Feature Importance)")
    st.markdown("Understand which features drive the model's predictions.")
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    st.subheader("Feature Importance (Random Forest)")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances[indices][:10], y=X.columns[indices][:10], ax=ax_fi, palette="viridis")
    ax_fi.set_title("Top 10 Most Important Features")
    st.pyplot(fig_fi)
    
    st.subheader("SHAP Summary Plot")
    st.markdown("SHAP values show the impact of each feature on the model's output for individual predictions.")
    with st.spinner("Calculating SHAP values..."):
        # SHAP can be slow, so we use a sample
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        
        fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
        # SHAP 0.50.0 handling
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(fig_shap)

with tab4:
    st.header("Custom Data Upload")
    st.markdown("Upload your own dataset (CSV) to run predictions using the optimized model. The dataset must have the same feature columns as the Breast Cancer dataset.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", custom_df.head())
            
            best_model = joblib.load('best_svm_model.pkl')
            
            if st.button("Run Predictions on Uploaded Data"):
                predictions = best_model.predict(custom_df)
                custom_df['Predicted_Diagnosis'] = ['Malignant' if p == 1 else 'Benign' for p in predictions]
                st.write("Results:")
                st.dataframe(custom_df)
                
                # Option to download results
                csv = custom_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name='predictions_output.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error processing file: {e}. Please ensure it matches the required format.")
