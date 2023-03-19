
import tempfile
import tkinter as tk
import webbrowser
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def open_web_view(html_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        f.write(html_content.encode())
        f.flush()
        webbrowser.open(f.name)


def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def plot_charts(df):
    fig_pie = px.pie(df, names=df.columns[-1], title='Pie Chart of Target Variable')
    fig_scatter = px.scatter_matrix(df, dimensions=df.columns[:-1], color=df.columns[-1],
                                    title='Scatter Matrix of Features')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    mse_var.set(f"MSE: {mse:.2f}")
    r2_var.set(f"R2 Score: {r2:.2f}")

    fig_regression = make_subplots(rows=1, cols=X.shape[1], subplot_titles=X.columns)
    for idx, col in enumerate(X.columns):
        fig_regression.add_trace(go.Scatter(x=X[col], y=y, mode='markers', name=col, showlegend=False), row=1,
                                 col=idx + 1)
        X_reg = X[[col]]
        model_reg = train_model(X_reg, y)
        y_reg = model_reg.predict(X_reg)
        fig_regression.add_trace(go.Scatter(x=X[col], y=y_reg, mode='lines', name=f"{col}_reg", showlegend=False),
                                 row=1, col=idx + 1)

    fig_regression.update_layout(title_text='Linear Regression Plots', showlegend=False)

    fig_histograms = make_subplots(rows=1, cols=X.shape[1], subplot_titles=X.columns)
    for idx, col in enumerate(X.columns):
        fig_histograms.add_trace(go.Histogram(x=X[col], name=col), row=1, col=idx + 1)

    fig_histograms.update_layout(title_text='Histograms')

    open_web_view(pio.to_html(fig_pie, full_html=False))
    open_web_view(pio.to_html(fig_scatter, full_html=False))
    open_web_view(pio.to_html(fig_regression, full_html=False))
    open_web_view(pio.to_html(fig_histograms, full_html=False))


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    file_path_var.set(file_path)


def analyze_data():
    file_path = file_path_var.get()
    df = load_dataset(file_path)
    plot_charts(df)


root = tk.Tk()
root.title("Data Analysis")

file_path_var = tk.StringVar()
mse_var = tk.StringVar()
r2_var = tk.StringVar()

ttk.Label(root, text="Select CSV file:").grid(column=0, row=0)
ttk.Entry(root, textvariable=file_path_var, width=50).grid(column=1, row=0)
ttk.Button(root, text="Browse", command=browse_file).grid(column=2, row=0)
ttk.Button(root, text="Analyze", command=analyze_data).grid(column=1, row=1)
ttk.Label(root, textvariable=mse_var).grid(column=1, row=2)
ttk.Label(root, textvariable=r2_var).grid(column=1, row=3)

mse_explanation = "MSE (Mean Squared Error) measures the average squared difference between predicted and actual values. Lower values indicate better model performance."
r2_explanation = "R2 Score represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, with higher values indicating better model performance."

ttk.Label(root, text=mse_explanation, wraplength=400).grid(column=1, row=4)
ttk.Label(root, text=r2_explanation, wraplength=400).grid(column=1, row=5)

root.mainloop()
