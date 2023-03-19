import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

# Train logistic regression model
def train_model(X_train, y_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, report, confusion

# Plot charts
def plot_charts(X, y):
    pie_data = y.value_counts()
    fig_pie = px.pie(pie_data, values='Outcome', names=pie_data.index, title='Distribution of Outcomes')
    fig_scatter = px.scatter_matrix(X, dimensions=X.columns, color=y, title='Scatter Matrix of Features')

    fig_histograms = make_subplots(rows=2, cols=4, subplot_titles=X.columns)
    for idx, col in enumerate(X.columns, start=1):
        fig_histograms.add_trace(go.Histogram(x=X[col], nbinsx=20, name=col), row=(idx - 1) // 4 + 1,
                                 col=((idx - 1) % 4) + 1)
    fig_histograms.update_layout(title_text='Histograms of Features', showlegend=False)

    pio.write_html(fig_pie, file='pie_chart.html', auto_open=False)
    pio.write_html(fig_scatter, file='scatter_matrix.html', auto_open=False)
    pio.write_html(fig_histograms, file='histograms.html', auto_open=False)

# Browse file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    file_path_var.set(file_path)

# Load and analyze data
def analyze_data():
    file_path = file_path_var.get()
    X, y = load_dataset(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy, report, confusion = evaluate_model(model, X_test, y_test)

    accuracy_var.set(f"Accuracy: {accuracy * 100:.2f}%")
    report_var.set(report)
    confusion_var.set(f"Confusion Matrix:\n{confusion}")

    plot_charts(X, y)

# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction")

file_path_var = tk.StringVar()
accuracy_var = tk.StringVar()
report_var = tk.StringVar()
confusion_var = tk.StringVar()

ttk.Label(root, text="Select CSV file:").grid(column=0, row=0)
ttk.Entry(root, textvariable=file_path_var, width=50).grid(column=1, row=0)
ttk.Button

(root, text="Browse", command=browse_file).grid(column=2, row=0)
ttk.Button(root, text="Analyze", command=analyze_data).grid(column=1, row=1)
ttk.Label(root, textvariable=accuracy_var).grid(column=1, row=2)
ttk.Label(root, textvariable=report_var).grid(column=1, row=3)
ttk.Label(root, textvariable=confusion_var).grid(column=1, row=4)

root.mainloop()

