import faicons as fa
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression
from shared import app_dir
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

# Load the correct CSV file
tips = pd.read_csv(app_dir / "time_series.csv", parse_dates=["date"])
tips.set_index("date", inplace=True)

# Agregar la columna 'day_of_year' al DataFrame tips
tips['day_of_year'] = tips.index.dayofyear

# Change variable names according to your CSV columns
ventas_rng = (tips.ventas.min(), tips.ventas.max())

# Seleccionar el rango de fechas de noviembre
ultimos_dias_noviembre_2018 = tips.loc["2018-11-24":"2018-11-30"]

# Calculate the monthly average for each year
monthly_avg = tips.resample('ME').ventas.mean()

# Prepare the data for the model
summer_data = tips.loc[(tips.index.month.isin([6, 7, 8, 9])) & (tips.index.year.isin([2017, 2018]))]
summer_data.loc[:, 'day_of_year'] = summer_data.index.dayofyear

# Features and labels for the model
X = summer_data[['day_of_year']]
y = summer_data['ventas']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict sales for a specific date range
def predict_sales(start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    day_of_year = dates.dayofyear
    predictions = model.predict(day_of_year.values.reshape(-1, 1))
    return pd.DataFrame({'Fecha': dates, 'Ventas_Predichas': predictions}, index=dates)

# Predict sales for summer 2019
predicted_sales_2019 = predict_sales('2019-06-01', '2019-09-30')

# Icons setup
ICONS = {
    "user": fa.icon_svg("user", "regular"),
    "wallet": fa.icon_svg("wallet"),
    "currency-dollar": fa.icon_svg("dollar-sign"),
    "ellipsis": fa.icon_svg("ellipsis"),
}

# UI configuration
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider(
            "ventas",
            "Rango de Ventas",
            min=ventas_rng[0],
            max=ventas_rng[1],
            value=ventas_rng,
            pre="$",
        ),
        ui.input_action_button("reset", "Resetear Filtros"),
        open="desktop",
    ),
    ui.layout_columns(
        ui.value_box(
            "Número de Registros", ui.output_ui("num_records"), showcase=ICONS["user"]
        ),
        ui.value_box(
            "Ventas Promedio", ui.output_ui("average_sales"), showcase=ICONS["wallet"]
        ),
        ui.value_box(
            "Desviación Estándar", ui.output_ui("std_dev"), showcase=ICONS["currency-dollar"]
        ),
        ui.value_box(
            "Pronóstico Verano 2019", ui.output_ui("summer_2019_forecast"), showcase=ICONS["ellipsis"]
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Datos de Ventas"), ui.output_data_frame("table"), full_screen=True
        ),
        ui.card(
            ui.card_header("Ventas Mensuales por Año"),
            output_widget("monthly_sales_plot"),
            full_screen=True
        ),
        ui.card(
            ui.card_header("Ventas Diarias en Junio y Julio"),
            output_widget("summer_sales_plot"),
            full_screen=True
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Predicción de Ventas Verano 2019"),
            output_widget("predicted_summer_sales_plot"),
            full_screen=True
        ),
        ui.card(
            ui.card_header("Comparación"), ui.output_data_frame("comparison_sales_table"), full_screen=True,
        ),
        ui.card(
            ui.card_header("Ventas Diarias de Todos los Años"),
            output_widget("all_years_sales_plot"),
            full_screen=True
        ),
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Predicción Primera Semana Diciembre 2018"),
            output_widget("noviembre_diciembre_sales_plot"), full_screen=True,
        ),
        ui.card(
            ui.card_header("Tabla de prediccion de Ventas Diarias Primera Semana Diciembre 2018"),
            ui.output_data_frame("noviembre_diciembre_sales_table"), full_screen=True,
        ),
    ),
    title="Análisis de Ventas",
    fillable=True,

)

# Agrega estilos CSS para el contenedor principal
app_ui = ui.div(
    app_ui,
    style="max-height: 100vh; overflow-y: auto;"
)

# Server logic
def server(input, output, session):
    @reactive.calc
    def filtered_data():
        ventas = input.ventas()
        return tips[tips.ventas.between(ventas[0], ventas[1])]

    @render.ui
    def num_records():
        return filtered_data().shape[0]

    @render.ui
    def average_sales():
        return f"${filtered_data().ventas.mean():.2f}"

    @render.ui
    def std_dev():
        return f"${filtered_data().ventas.std():.2f}"

    @render.ui
    def summer_2019_forecast():
        if not predicted_sales_2019.empty:
            return f"Predicción: ${predicted_sales_2019['Ventas_Predichas'].mean():.2f}"
        else:
            return "No hay datos suficientes para pronóstico"

    @render.data_frame
    def table():
        df = filtered_data().reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')  # Format the date as YYYY-MM-DD
        return df[['date', 'ventas']]

    @render_plotly
    def monthly_sales_plot():
        df = monthly_avg.to_frame().reset_index()
        df['year'] = df['date'].dt.year
        fig = px.line(df, x='date', y='ventas', color='year', title='Ventas Mensuales por Año')
        fig.update_traces(mode='lines+markers')
        return fig

    @render_plotly
    def summer_sales_plot():
        df_2017 = tips.loc["2017-06-01":"2017-07-31"]
        df_2018 = tips.loc["2018-06-01":"2018-07-31"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df_2017.index, y=df_2017['ventas'], mode='lines+markers', name='Verano 2017')
        )
        fig.add_trace(
            go.Scatter(x=df_2018.index, y=df_2018['ventas'], mode='lines+markers', name='Verano 2018')
        )
        fig.update_layout(
            title='Ventas Diarias en Junio y Julio para 2017 y 2018',
            xaxis_title='Fecha',
            yaxis_title='Ventas',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @render_plotly
    def predicted_summer_sales_plot():
        fig = px.line(predicted_sales_2019.reset_index(), x='Fecha', y='Ventas_Predichas',
                      title='Predicción de Ventas Verano 2019')
        fig.update_traces(mode='lines+markers')
        return fig

    @render.data_frame
    def comparison_sales_table():
        summer_2017 = tips['2017-06-01':'2017-09-30'].resample('ME').mean()['ventas']
        summer_2018 = tips['2018-06-01':'2018-09-30'].resample('ME').mean()['ventas']
        summer_2019 = predicted_sales_2019.resample('ME').mean()['Ventas_Predichas']
        months = ['Junio', 'Julio', 'Agosto', 'Septiembre']
        month_names = ['Junio', 'Julio', 'Agosto', 'Septiembre']
        comparison_df = pd.DataFrame({
            'Mes': month_names,
            '2017': summer_2017.values,
            '2018': summer_2018.values,
            '2019': summer_2019.values
        })
        return comparison_df

    # Corrección de la función para la tabla de noviembre y diciembre
    ultimos_dias_noviembre_2018 = tips.loc["2018-11-24":"2018-11-30"]

    # Preparar los datos de entrenamiento para el modelo de regresión lineal de diciembre
    X_noviembre = ultimos_dias_noviembre_2018[['day_of_year']]
    y_noviembre = ultimos_dias_noviembre_2018['ventas']

    # Entrenar un nuevo modelo de regresión lineal con los datos de noviembre
    model_diciembre = LinearRegression()
    model_diciembre.fit(X_noviembre, y_noviembre)

    # Función para predecir las ventas para la primera semana de diciembre
    def predict_sales_diciembre(start_date, end_date):
        dates = pd.date_range(start_date, end_date)
        day_of_year = dates.dayofyear
        predictions = model_diciembre.predict(day_of_year.values.reshape(-1, 1))
        return pd.DataFrame({'Fecha': dates, 'Ventas_Predichas': predictions}, index=dates)

    # Generar la predicción para la primera semana de diciembre
    prediccion_primera_semana_diciembre_2018 = predict_sales_diciembre("2018-12-01", "2018-12-07")

    @render_plotly
    def noviembre_diciembre_sales_plot():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=ultimos_dias_noviembre_2018.index, y=ultimos_dias_noviembre_2018['ventas'],
                       mode='lines+markers', name='Ventas Noviembre')
        )
        fig.add_trace(
            go.Scatter(x=prediccion_primera_semana_diciembre_2018.index,
                       y=prediccion_primera_semana_diciembre_2018['Ventas_Predichas'], mode='lines+markers',
                       name='Predicción Diciembre')
        )
        fig.update_layout(
            title='Predicción Primera Semana Diciembre 2018',
            xaxis_title='Fecha',
            yaxis_title='Ventas',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    @render.data_frame
    def noviembre_diciembre_sales_table():
        ultimos_dias_noviembre_2018_filled = ultimos_dias_noviembre_2018.fillna(0)
        # Crear un nuevo DataFrame con los datos de ventas de noviembre
        df_with_prediction = pd.DataFrame({
            'Fecha': prediccion_primera_semana_diciembre_2018.index.strftime('%Y-%m-%d'),
            'Ventas_Predichas': prediccion_primera_semana_diciembre_2018['Ventas_Predichas'].astype(int)
        })
        return df_with_prediction

    # Crear un nuevo DataFrame para el pronóstico de diciembre
    pronostico_diciembre_df = pd.DataFrame({
        'Fecha': prediccion_primera_semana_diciembre_2018.index,
        'Ventas_Predichas': prediccion_primera_semana_diciembre_2018['Ventas_Predichas'].astype(int)
    })

    # Guardar el pronóstico de diciembre en un archivo CSV
    pronostico_diciembre_df.to_csv(app_dir / "pronostico_diciembre.csv", index=False)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_slider("ventas", value=ventas_rng)

    @render_plotly
    def all_years_sales_plot():
        # Reset the index to make 'date' a column again if it's set as an index
        df = tips.reset_index()
        # Ensure the date column is treated as datetime type
        df['date'] = pd.to_datetime(df['date'])
        # Extract the year from the date for coloring purposes
        df['year'] = df['date'].dt.year

        # Use the entire dataset for this plot and color by 'year'
        fig = px.line(df, x='date', y='ventas', color='year', title='Ventas Diarias de Todos los Años',
                      labels={'year': 'Año'},  # This changes the legend title from 'year' to 'Año'
                      line_shape='linear',  # You can also choose 'spline' for smoother lines
                      markers=True)  # Includes markers at data points

        # Optional: Update the layout if needed
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Ventas",
            legend_title="Año",
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

app = App(app_ui, server)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)