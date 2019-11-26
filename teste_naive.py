import pandas as pd
import numpy as np

# Classe para definicoes das funcoes do modelo de classificacao Naive Bayes
class Naive:
	# funcao de inicializacao recebe os conjuntos de dados e o taxa de amostragem para teste (complementar a de treinamento)
	def __init__(self, X, y, test_size=0.1, alpha=1, n_test=10):
		self.X = X
		self.y = y
		self.test_size = test_size
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.alpha = alpha
		self.n_test = n_test
		self.train_test_split()


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
		y_train = np.delete(np.array(self.y), test_list, axis=0)
		
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

	def class_prob(self):
		classes, counts = np.unique(self.y_train, return_counts=True)
		dic = dict(zip(classes, counts))
		
		for key in dic.keys():
			dic[key] = dic[key] / len(self.y_train)
		return dic


	# definicao das probabilidades de cada classe ocorrer
	def predict(self, query):
		class_dic = self.class_prob()
		log_P_h_D = np.zeros(len(class_dic))

		for classe in range(len(class_dic)):
			log_P_h_D[classe] = np.log(class_dic[list(class_dic.keys())[classe]])
			for atr in range(self.X_train.shape[1]):
				nom_atr = query[atr]
				ids = np.where(self.y_train == list(class_dic.keys())[classe])
				P_nom_h = (np.sum((self.X_train[ids])[:,atr] == nom_atr) + self.alpha) / len(ids[0])
				if P_nom_h != 0:
					log_P_h_D[classe] = log_P_h_D[classe] + np.log(P_nom_h)


		P_h_D = np.exp(log_P_h_D)
		P_h_D = P_h_D/np.sum(P_h_D)

		return class_dic.keys(), P_h_D 

	def score(self):
		score = 0
		count_right_total = 0
		for i in range(self.n_test):
			self.train_test_split()
			count_right = 0
			for q in range(len(self.X_test)):
				pred = self.predict(self.X_test[q])
				classes = list(pred[0])
				proba = list(pred[1])
				correto = self.y_test[q]
				if classes[proba.index(np.max(proba))] == correto:
					count_right = count_right + 1
			count_right_total = count_right_total + count_right
		return count_right_total / len(self.X_test) * self.n_test
			


#main
# Leitura do dataset e definicao dos conjuntos de dados X e y
data = pd.read_csv("data_discretizado.csv")
y = data["rating"]
X = data.drop(["Unnamed: 0","Unnamed: 0.1","rating","title"],axis=1) #remocao das colunas que nao serao validas na classificacao

for i in range(20):
	n = Naive(X, y, alpha=20/10)
	print("alpha: " + str(n.alpha))
	print("score: " + str(n.score()))		




