
"""
@author__ = "Juan Francisco Illan"
@license__ = "GPL"
@version__ = "1.0.1"
@email__ = "juanfrancisco.illan@gmail.com"
"""

from flask import Flask, request, render_template
from flask_debugtoolbar import DebugToolbarExtension
import sqlite3
import matplotlib.pyplot as plt
import random
import pandas as pd

from classes import *
from blast_process import *
from pyTrader import *

app = Flask(__name__)
app.config['SECRET_KEY'] = "lailolailo"
app.debug = True
toolbar = DebugToolbarExtension(app)

# Controller to index
@app.route('/')
def home():
	return render_template('index.html')




@app.route('/run', methods=["GET"])
def reload_data():

	# Read the CSV file
	stock_data = pd.read_csv("IBEX_2017-2020.csv")
	stock_data_test = pd.read_csv("IBEX_2020_2023.csv")

	# View the first 5 rows
	stock_data.head()


	x_datetime = stock_data_test.iloc[:,0:1].values
	#x_datetime = pd.to_datetime(x_datetime, format="%Y-%m-%d") # # Primera columna
	y_stock_data_test = stock_data_test.iloc[:,1:2]


	# valores indicados en el formulario 
	#cbp = ConfigBlastPathogen()
	#cbp.querry_seq = request.form['querry_seq'] #The querry sequence we search for
	#cbp.mode = int(request.form['mode']) # Mode of execution

	#if (app.tokenizerLSTM == None or app.modelLSTM == None) :
	#	seq_data =  pd.read_csv('database/pathogen_sars_cov_2.csv', sep=';', engine='python')		
	#	app.tokenizerLSTM, app.modelLSTM , app.statsLSTM, app.shortModelSummary = createClasiffierLSTM(seq_data)

	prediccion = createAdvisorLSTM(stock_data,stock_data_test,stock_data_test)

	## mode process the secuences		
	#blastResultPathogen, statsPrediction = clasiffierLSTM(cbp, app.tokenizerLSTM, app.modelLSTM)	
		
	# response html with blastResult
	#return render_template('blast_pathogen.html', configBlastPathogen=cbp, blastResultPathogen=blastResultPathogen, statsLSTM=app.statsLSTM, shortModelSummary = app.shortModelSummary, statsPrediction=statsPrediction)
	
	
	#dataStats_predic = {'stock_data_test':y_stock_data_test,
	#			'prediccion':prediccion}	
	
	#dataStats_predic = dataStats_predic.set_index(x_datetime)
						

	plt.xlabel('datetime')
	plt.ylabel('stock price')

	#plt.plot(,)
	plt.plot(y_stock_data_test, label = "line 1")
	plt.gcf().autofmt_xdate()
	plt.show()

	plt.plot(prediccion, label = "line 2")

	plt.gcf().autofmt_xdate()
	plt.show()

	hash = random.getrandbits(32)
	plt.savefig('static/images/temp/plot_' + str(hash) + '.png') 
	plt.close()  

	# response html with plotimage
	#return render_template('plot.html',plotImage='plot_' + str(hash) + '.png', filterPlot=filterPlot)

	return render_template('index.html',plotImage='plot_' + str(hash) + '.png')





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
	app.classifierMNB = None
	app.countVectorizer = None
	app.tokenizerLSTM = None
	app.modelLSTM = None
	app.run(host='0.0.0.0')

