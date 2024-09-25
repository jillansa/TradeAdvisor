
"""
@author__ = "Juan Francisco Illan"
@license__ = "GPL"
@version__ = "1.0.1"
@email__ = "juanfrancisco.illan@gmail.com"
"""

from flask import Flask, request, render_template, jsonify
from flask_debugtoolbar import DebugToolbarExtension
import sqlite3
import matplotlib.pyplot as plt
import random
import pandas as pd
import csv
from datetime import datetime, timedelta

from classes import *
from pyTrader import *

import numpy as np

import requests
import yfinance as yf
import pandas_ta as ta


app = Flask(__name__)
app.config['SECRET_KEY'] = "lailolailo"
app.debug = True
toolbar = DebugToolbarExtension(app)


################################################################################
################################################################################
################################################################################

# Controller to index
@app.route('/', methods=["GET"])
def home():

	tiempo = []
	pib_usa = []
	tipos_usa = []
	inflaccion_usa = []
	sp500_usa = []
	gold = []

	csv_USA = '.\\database\\usa.csv'

	# Abrir el archivo CSV y leer los datos
	with open(csv_USA, newline='') as csvfile:
		# Crear un objeto lector de CSV
		lector_csv = csv.reader(csvfile, delimiter=';')
		int = 1
		# Iterar sobre cada fila del CSV
		for fila in lector_csv:
			if int == 1:
				int = 0
				continue
			# Añadir el dato de la segunda columna a la lista
			tiempo.append(float(fila[0].replace(',', '.')))
			pib_usa.append(float(fila[1].replace(',', '.')))  # Indice 1 para la segunda columna - PIB
			tipos_usa.append(float(fila[2].replace(',', '.')))  # TIPOS
			inflaccion_usa.append(float(fila[3].replace(',', '.')))  # INFLACCION
			sp500_usa.append(float(fila[4].replace(',', '.')))  # INFLACCION
			gold.append(float(fila[5].replace(',', '.')))  # INFLACCION

	fig, ax1 = plt.subplots()

	# Crear la gráfica
	plt.plot(tiempo, pib_usa, label='PIB USA')
	plt.plot(tiempo, tipos_usa, label='Tipos de Interés USA')
	plt.plot(tiempo, inflaccion_usa, label='Inflación USA')

	#color = 'tab:red'
	#ax1.set_xlabel('Tiempo')
	#ax1.set_ylabel('(%)')
	#ax1.plot(tiempo, pib_usa, color = "red")
	#ax1.plot(tiempo, tipos_usa, color = "blue")
	#ax1.plot(tiempo, inflaccion_usa, color = "green")
	#ax1.tick_params(axis='y')

	#plt.legend()


	# Crear segundo eje y para los activos de mercado
	#ax2 = ax1.twinx()
	#color = 'tab:black'
	#ax2.set_ylabel('Cotizacion Activo (%)')
	#ax2.plot(tiempo, sp500_usa, color="black")
	#ax2.plot(tiempo, gold, color="yellow")
	#ax2.tick_params(axis='y', labelcolor=color)

	#plt.plot(tiempo, sp500_usa, label='SP500')
	#plt.plot(tiempo, gold, label='Gold')

	plt.xlabel('Tiempo')
	plt.ylabel('%')

	#fig.tight_layout()	
	plt.legend()

	hash = random.getrandbits(32)	
	plt.savefig('static/images/temp/plot_' + str(hash) + '.png') 
	plt.close() 

	# Mostrar la gráfica
	#plt.show()

	# response html
	return render_template('index.html', plotImage='plot_' + str(hash) + '.png')
	#return render_template('index.html')


################################################################################
################################################################################
################################################################################


# Controller to index
@app.route('/generarGrafica', methods=["POST"])
def generarGrafica():

	data = request.get_json()
	variables = data.get('variables', [])

	tiempo = []
	pib_usa = []
	tipos_usa = []
	inflaccion_usa = []
	sp500_usa = []
	gold = []

	csv_USA = '.\\database\\usa.csv'

	# Abrir el archivo CSV y leer los datos
	with open(csv_USA, newline='') as csvfile:
		# Crear un objeto lector de CSV
		lector_csv = csv.reader(csvfile, delimiter=';')
		int = 1
		# Iterar sobre cada fila del CSV
		for fila in lector_csv:
			if int == 1:
				int = 0
				continue
			# Añadir el dato de la segunda columna a la lista
			tiempo.append(float(fila[0].replace(',', '.')))
			pib_usa.append(float(fila[1].replace(',', '.')))  # Indice 1 para la segunda columna - PIB
			tipos_usa.append(float(fila[2].replace(',', '.')))  # TIPOS
			inflaccion_usa.append(float(fila[3].replace(',', '.')))  # INFLACCION
			sp500_usa.append(float(fila[4].replace(',', '.')))  # INFLACCION
			gold.append(float(fila[5].replace(',', '.')))  # INFLACCION

	fig, ax1 = plt.subplots()

	# Crear la gráfica
	plt.plot(tiempo, pib_usa, label='PIB USA')
	plt.plot(tiempo, tipos_usa, label='Tipos de Interés USA')
	plt.plot(tiempo, inflaccion_usa, label='Inflación USA')

	#color = 'tab:red'
	#ax1.set_xlabel('Tiempo')
	#ax1.set_ylabel('(%)')
	#ax1.plot(tiempo, pib_usa, color = "red")
	#ax1.plot(tiempo, tipos_usa, color = "blue")
	#ax1.plot(tiempo, inflaccion_usa, color = "green")
	#ax1.tick_params(axis='y')

	#plt.legend()


	# Crear segundo eje y para los activos de mercado
	#ax2 = ax1.twinx()
	#color = 'tab:black'
	#ax2.set_ylabel('Cotizacion Activo (%)')
	#ax2.plot(tiempo, sp500_usa, color="black")
	#ax2.plot(tiempo, gold, color="yellow")
	#ax2.tick_params(axis='y', labelcolor=color)

	#plt.plot(tiempo, sp500_usa, label='SP500')
	#plt.plot(tiempo, gold, label='Gold')

	plt.xlabel('Tiempo')
	plt.ylabel('%')

	#fig.tight_layout()	
	plt.legend()

	hash = random.getrandbits(32)	
	plt.savefig('static/images/temp/plot_' + str(hash) + '.png') 
	plt.close() 

	# Mostrar la gráfica
	#plt.show()

	# response html
	#return render_template('index.html', plotImage='plot_' + str(hash) + '.png')
	# Devuelve la URL de la imagen generada como respuesta
	return jsonify({'imageUrl': 'http://localhost:5000/static/images/temp/plot_' + str(hash) + '.png'})
	#return render_template('index.html')


################################################################################
################################################################################
################################################################################

symbols = ['^DJI', 'NQ=F', '^GSPC', '^RUT', '^IBEX', '^STOXX50E', 'EURUSD=X', 'GC=F', 'SI=F', 'CL=F', '^FVX', '^TYX', 'GE', 'MCD']
	# Dow Jones Industrial Average (^DJI)
	# NQ=F : NASDAQ
	# ^GSPC : SP500
	# Russell 2000 (^RUT)
	# ^IBEX : IBEX35
	# ^STOXX50E : Eurostock50
	# FTSE 100 (^FTSE)
	# Nikkei 225 (^N225)
	# EURUSD=X : EUR/USD
	# GBP/USD (GBPUSD=X)
	# USD/JPY (JPY=X)
	# Gold Apr 24 (GC=F)
	# Silver May 24 (SI=F)
	# Crude Oil May 24 (CL=F)
	# 13 WEEK TREASURY BILL (^IRX)
	# Treasury Yield 5 Years (^FVX)
	# Treasury Yield 30 Years (^TYX)
	# Bitcoin USD (BTC-USD)
	# General Electric Company (GE)
	# McDonald's Corporation (MCD)

	# ^ = %5E


################################################################################
################################################################################
################################################################################


@app.route('/download', methods=["POST"])
def download():
	# Descargar 
	url = 'https://query1.finance.yahoo.com/v7/finance/download/SYMBOL?period1=86400&period2=1900240905&interval=1d&events=history&includeAdjustedClose=true&crumb=wANR/eluUjE'

	# ALTERNATIVE: 
	# yfinance 0.2.37 : Download market data from Yahoo! Finance's API
	# https://pypi.org/project/yfinance/

	# Define the stock symbol and timeframe
	# symbol = 'AAPL' # Apple stock
	# end_date = datetime.today()
	# start_date = end_date - timedelta(days=120)  # 4 months before today
	# stock_data = yf.download(symbol, start=start_date, end=end_date)


	#NASDAQ Composite: %5EIXIC
	#IBEX: %5EIBEX
	#EUROSTOCK: %5ESTOXX50E
	#EURUSD: EURUSD=X

	# Encabezado de usuario simulando un navegador Chrome en Windows 10
	headers = {
    	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
	}

	#symbols = ['%5EIXIC', '%5EIBEX', '%5ESTOXX50E', 'EURUSD=X']

	for syn in symbols:

		response = requests.get(url.replace('SYMBOL',syn), headers=headers)
		if response.status_code == 200:
			# Nombre del archivo destino en disco
			nombre_archivo = "database/"+syn + '.csv'
			
			# Abre un archivo en modo de escritura ('w') y escribe el contenido descargado
			with open(nombre_archivo, 'w') as f:
				f.write(response.content.decode().replace(',', ';').replace('.', ',')) # To open in EXCEL 
			
			print(f'El archivo {nombre_archivo} ha sido descargado y guardado correctamente.')
		else:
			print('La solicitud para descargar el archivo ha fallado.')
			
	return render_template('index.html')
			
################################################################################
################################################################################
################################################################################


@app.route('/run_lstm', methods=["POST"])
def run_lstm():


	# Read the CSV file
	stock_data = pd.read_csv("database/^IBEX.csv", delimiter=';', decimal=",")
	#stock_data_test = pd.read_csv("database/IBEX_2020_2023.csv", delimiter=';', decimal=",")	
	
	# Create date column to datatime type object
	stock_data['Datetime'] = pd.to_datetime(stock_data['Date'],format='%Y-%m-%d')

	# Eliminar filas con valores faltantes en 'Close' 
	stock_data = stock_data.dropna(subset=['Close'])
	stock_data = stock_data[stock_data['Close'] != 'null']
	
	# DiffClose de la columna actual y la columna de la fila anterior
	stock_data['DiffClose'] = stock_data['Close'] - stock_data['Close'].shift(1)
	stock_data['DiffClose(%)'] = stock_data['DiffClose'] * 100 / stock_data['Close']
	

	# Calculate technical indicators using pandas-ta
	stock_data.ta.sma(length=20, append=True)  # Calcular SMA de 20 períodos y agregarlo al DataFrame
	stock_data.ta.ema(length=50, append=True)
	stock_data.ta.bbands(length=20, append=True)  # Calcular Bandas de Bollinger y agregarlas al DataFrame
	stock_data.ta.stoch(append=True)  # Calcular el Oscilador Estocástico y agregarlo al DataFrame
	stock_data.ta.rsi(length=14, append=True)  # Calcular RSI con un período de 14 y agregarlo al DataFrame	
	stock_data.ta.macd(append=True)
	stock_data.ta.adx(append=True)
	#stock_data.ta.willr(append=True)
	#stock_data.ta.cmf(append=True)
	#stock_data.ta.psar(append=True)

	# Puedes continuar agregando más indicadores según tus necesidades
	# https://ta-lib.org/
	# Use TA-Lib to add technical analysis to your own financial market trading applications
	# 200 indicators such as ADX, MACD, RSI, Stochastic, Bollinger Bands etc...  See complete list...
	# Candlestick patterns recognition
	# Core written in C/C++ with API also available for Python.

	# Calcula el número de filas para el conjunto de entrenamiento (70%)
	n_train = int(0.7 * len(stock_data))

	# Divide el DataFrame en dos partes: entrenamiento y prueba
	stock_data_train = stock_data.iloc[:n_train]  # Primer 70% de los datos
	stock_data_test = stock_data.iloc[n_train:]   # Restante 30% de los datos

	# Ahora tienes stock_data_train y stock_data_test como pandas DataFrames con el 70% y 30% de los datos
		
	# View the first 5 rows
	stock_data_train.head()

	# To CSV FileSystem
	stock_data_train.to_csv('database/^IBEX_train_IA.csv', index=False, sep=';', decimal=",")  # index=False para evitar escribir el índice del DataFrame en el archivo CSV
	stock_data_test.to_csv('database/^IBEX_test_IA.csv', index=False, sep=';', decimal=",")  # index=False para evitar escribir el índice del DataFrame en el archivo CSV

	# Metodo
	time_step = 10  # el contexto en la toma de posiciones es de 10 sesiones 
	
	# Variable Inputs
	variableInput = 'Open'  # DiffClose(%)	
	variableInput = 'DiffClose(%)'  # DiffClose(%)	
	
	# Variable Outputs
	variableOutput = 'Open'  # DiffClose(%)
	variableOutput = 'DiffClose(%)'  # DiffClose(%)

	app.modelLSTM, prediccion = createAdvisorMultivariateLSTM(stock_data_train,stock_data_test, time_step, variableInput, variableOutput, True)
	
	y_stock_data_test = stock_data_test[variableOutput]

	plt.xlabel('Datetime')
	plt.ylabel(variableOutput + ' stock price')

	# mostramos los datos ce cierre de sesion reales del conjunto de test
	# quitamos los 10 primeros valores para los que no tenemos prediccion
	plt.plot(stock_data_test['Datetime'][time_step+1:], y_stock_data_test[time_step+1:], label = variableOutput + " stock price")	
	plt.plot(stock_data_test['Datetime'][time_step+1:], prediccion, label = "prediction")

	plt.gcf().autofmt_xdate()
	plt.legend()

	plt.show()

	#hash = random.getrandbits(32)
	#plt.savefig('static/images/temp/plot_' + str(hash) + '.png') 
	#plt.close()  

	# response html with plotimage
	return render_template('index.html')
	#return render_template('index.html',plotImage='plot_' + str(hash) + '.png')


################################################################################
################################################################################
################################################################################

@app.route('/run_lstm_example', methods=["GET"])
def run_lstm_example():

	# Read the CSV file
	stock_data = pd.read_csv("database/IBEX_2017-2020.csv", delimiter=';', decimal=",")	
	stock_data_test = pd.read_csv("database/IBEX_2020_2023.csv", delimiter=';', decimal=",")	

	#Convert date column to datatime type object
	stock_data['Datetime'] = pd.to_datetime(stock_data['Date'],format='%d/%m/%Y')
	stock_data.drop(columns=['Date'])
	stock_data_test['Datetime'] = pd.to_datetime(stock_data_test['Date'],format='%d/%m/%Y')
	stock_data_test.drop(columns=['Date'])


	# View the first 5 rows
	stock_data.head()

	# Metodo
	time_step = 10  # el contexto en la toma de posiciones es de 10 sesiones 
	variableInput = 'Close'  # 
	variableOutput = 'Close'  # 
	

	app.modelLSTM, prediccion = createAdvisorLSTM(stock_data,stock_data_test, time_step, variableInput, variableOutput)

	y_stock_data_test = stock_data_test[variableOutput]

	plt.xlabel('Datetime')
	plt.ylabel(variableOutput + ' stock price')

	# mostramos los datos ce cierre de sesion reales del conjunto de test
	# quitamos los 10 primeros valores para los que no tenemos prediccion
	plt.plot(stock_data_test['Datetime'][time_step+1:], y_stock_data_test[time_step+1:], label = variableOutput + " stock price")	
	plt.plot(stock_data_test['Datetime'][time_step+1:], prediccion, label = "prediction")

	plt.gcf().autofmt_xdate()
	plt.legend()

	plt.show()

	hash = random.getrandbits(32)
	plt.savefig('static/images/temp/plot_' + str(hash) + '.png') 
	plt.close()  

	# response html with plotimage
	return render_template('index.html')
	#return render_template('index.html',plotImage='plot_' + str(hash) + '.png')


################################################################################
################################################################################
################################################################################



# Controller to blast access
@app.route('/about', methods=["GET"])
def about():

	return render_template('about.html')


################################################################################
################################################################################
################################################################################

# startup HTTP web service
if __name__ == '__main__':
	print("Running service ...")
	app.run(debug=True)
	app.modelLSTM = None
	app.run(host='0.0.0.0')

