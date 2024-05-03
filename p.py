import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar los datos
data = pd.read_csv('time_series.csv')
data['date'] = pd.to_datetime(data['date'])

# Estadísticas básicas
total_records = len(data)
mean_sales = data['ventas'].mean()
std_dev_sales = data['ventas'].std()

# Calcular promedios mensuales y preparar datos para gráficos
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
monthly_data = data.groupby(['year', 'month'])['ventas'].mean().unstack(0)

# Gráfico de promedios mensuales
plt.figure(figsize=(12, 6))
monthly_data.plot(kind='line', marker='o', colormap='viridis')
plt.title('Promedio Mensual de Ventas por Año')
plt.xlabel('Mes')
plt.ylabel('Ventas Promedio')
plt.grid(True)
plt.legend(title='Año')
plt.show()

# Visualización de ventas diarias en junio y julio
june_july_data = data[(data['month'] == 6) | (data['month'] == 7)]
plt.figure(figsize=(14, 7))
for year in june_july_data['year'].unique():
    subset = june_july_data[june_july_data['year'] == year]
    plt.plot(subset['date'], subset['ventas'], marker='o', label=f'Year {year}')
plt.title('Daily Sales in June and July by Year')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Pronóstico de ventas para la primera semana de diciembre de 2018
november_2018_data = data[(data['date'] >= '2018-11-24') & (data['date'] <= '2018-11-30')]
average_last_week_nov_2018 = november_2018_data['ventas'].mean()
december_first_week_2018 = [average_last_week_nov_2018] * 7  # Repetir para los 7 días de la semana
print('Predicted sales for the first week of December 2018:', december_first_week_2018)

# Preparar datos para el modelo de regresión
data['date_ordinal'] = data['date'].map(lambda x: x.toordinal())
X = data[['date_ordinal']]  # Feature
y = data['ventas']          # Target

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Generar fechas para el verano de 2019
summer_2019_dates = pd.to_datetime(['2019-06-01', '2019-07-01', '2019-08-01'])
summer_2019_ordinals = np.array([date.toordinal() for date in summer_2019_dates]).reshape(-1, 1)

# Hacer pronósticos para el verano de 2019
summer_2019_forecasts = model.predict(summer_2019_ordinals)
summer_2019_predictions = dict(zip(['June 2019', 'July 2019', 'August 2019'], summer_2019_forecasts))

# Visualización de la línea de tendencia y predicciones para el verano de 2019
plt.figure(figsize=(12, 6))
plt.scatter(data['date'], y, color='blue', label='Real data')
plt.plot(data['date'], model.predict(X), color='red', linewidth=2, label='Trend line')
plt.scatter(summer_2019_dates, summer_2019_forecasts, color='green', s=100, label='Summer 2019 Predictions')
plt.title('Sales Forecast for Summer 2019')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Imprimir las predicciones para el verano de 2019
print('Predicted sales for Summer 2019:', summer_2019_predictions)
