import pandas as pd
import numpy as np

# Classe para definicoes das funcoes do modelo de classificacao Naive Bayes
class Naive:
	# funcao de inicializacao recebe os conjuntos de dados e o taxa de amostragem para teste (complementar a de treinamento)
	def __init__(self, X, y, test_size=0.1):
		self.X = X
		self.y = y
		self.test_size = test_size
	# funcao para separacao dos indices, selecionados randomicamente, para teste e treino
	def train_test_split(self):
		# define o numero de elementos presentes no conjunto de teste
		n_test = np.int(np.floor(self.test_size * len(y.index)))

		# gera uma lista dos indices que estarao no conjunto de teste
		test_list = np.random.choice(len(self.y.index),n_test)
		
		# definicao dos conjuntos de teste
		X_test = np.array(self.X)[test_list]
		y_test = np.array(self.y)[test_list]
		
		# definicao do conjunto de treinamento - obtendo o complemento do conjunto de teste
		X_train = np.delete(np.array(self.X), test_list, axis=0)
		y_train = np.delete(np.array(self.X), test_list, axis=0)
		
		return X_train, y_train, X_test, y_test

	# definicao das probabilidades de cada classe ocorrer
	def fit(self):









#main
# Leitura do dataset e definicao dos conjuntos de dados X e y
data = pd.read_csv("data_discretizado.csv")
y = data["rating"]
X = data.drop(["Unnamed: 0","Unnamed: 0.1","rating","title"],axis=1) #remocao das colunas que nao serao validas na classificacao

n = Naive(X,y)
n.train_test_split()

