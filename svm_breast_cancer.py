import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv('breast-cancer.csv')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Pipeline and Model Benchmarking
    print("\n--- Model Benchmarking ---")
    pipelines = {
        "Logistic Regression": Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42))]),
        "Random Forest": Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=42))]),
        "SVM (RBF)": Pipeline([('scaler', StandardScaler()), ('classifier', SVC(kernel='rbf', random_state=42))])
    }

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        print(f"{name} Accuracy: {score:.4f}")

    # 3. SVM Hyperparameter Tuning with GridSearchCV
    print("\n--- SVM Hyperparameter Tuning ---")
    param_grid = {
        'classifier__C': [0.1, 1, 10], 
        'classifier__gamma': ['scale', 0.01, 0.1, 1],
        'classifier__kernel': ['rbf', 'linear']
    }
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(random_state=42))])
    grid = GridSearchCV(svm_pipeline, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    # 4. Evaluation and Confusion Matrix
    print("\n--- Best Model Evaluation ---")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Plot and save Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix - Best SVM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    plt.close()

    # 5. PCA and Decision Boundary Visualization
    print("\n--- Generating Decision Boundaries ---")
    # Need to scale and PCA for visualization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Train simplified models for 2D visualization
    svm_linear = SVC(kernel='linear', C=1.0).fit(X_pca, y)
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_pca, y)

    def plot_decision_boundary(model, X, y, title, filename):
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(title)
        
        # create legend
        handles, labels = scatter.legend_elements()
        if len(handles) > 0:
            plt.legend(handles, ['Benign', 'Malignant'][:len(handles)])
            
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

    plot_decision_boundary(svm_linear, X_pca, y, 'SVM Linear Kernel Decision Boundary (PCA)', 'svm_linear_pca.png')
    plot_decision_boundary(svm_rbf, X_pca, y, 'SVM RBF Kernel Decision Boundary (PCA)', 'svm_rbf_pca.png')
    print("\nAll tasks completed successfully!")

if __name__ == "__main__":
    main()
