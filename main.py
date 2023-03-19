import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import webbrowser


# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


# Train linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# Plot charts
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

    # Plot histograms
    fig_histograms = make_subplots(rows=1, cols=X.shape[1], subplot_titles=X.columns)
    for idx, col in enumerate(X.columns):
        fig_histograms.add_trace(go.Histogram(x=X[col], name=col), row=1, col=idx + 1)

    fig_histograms.update_layout(title_text='Histograms')

    pio.write_html(fig_pie, file='pie_chart.html', auto_open=False)
    pio.write_html(fig_scatter, file='scatter_matrix.html', auto_open=False)
    pio.write_html(fig_regression, file='linear_regression_plots.html', auto_open=False)
    # Save histograms to HTML file
    pio.write_html(fig_histograms, file='histograms.html', auto_open=False)

    webbrowser.open('pie_chart.html')
    webbrowser.open('scatter_matrix.html')
    webbrowser.open('linear_regression_plots.html')
    # Open histograms HTML file in default browser
    webbrowser.open('histograms.html')


# Browse file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    file_path_var.set(file_path)


# Load and analyze data
def analyze_data():
    file_path = file_path_var.get()
    df = load_dataset(file_path)
    plot_charts(df)


# Create the main window
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

root.mainloop()
