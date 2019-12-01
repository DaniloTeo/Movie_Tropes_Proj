import pandas as pd
import numpy as np

# Classe para definir as funcoes do ensemble de Naive-Bayes
class Ensemble:
	def __init__(self, X, y, test_size = 0.1, n_classifiers=5, type= "sum"):
		self.X = X
		self.y = y
		self.type = type
		self.n_classifiers = n_classifiers
		self.test_size = test_size
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.classifiers = []
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

	def fit(self):
		aux_classifiers = []
		dtype = [('class',Naive), ('score',np.float_)]
		for n in range(self.n_classifiers * 10):
			c = Naive(self.X, self.y, alpha = 0.2)
			aux_classifiers.append((c, c.score[0])) #VAI DAR ERRO AQUI, SCORE AGORA EH UMA ARRAY COM O SCORE NORMAL E O COM PESO
		aux_classifiers = np.array(aux_classifiers,dtype = dtype)
		aux_classifiers = sorted(aux_classifiers, key=lambda tup: tup[1], reverse=True)
		for i in range(self.n_classifiers):
			self.classifiers.append(aux_classifiers[i][0])

	def class_prob(self):
		classes, counts = np.unique(self.y_train, return_counts=True)
		dic = dict(zip(classes, counts))
		
		for key in dic.keys():
			dic[key] = dic[key] / len(self.y_train)
		return dic

	def predict(self, query, type=self.type):
		if type == "sum":
			classifications = []
			for classifier in self.classifiers:
				pred = classifier.predict(query)
				classes = pred[0]
				classifications.append(pred[1])
			pred = np.sum(classifications, axis=0) / self.n_classifiers
		elif type == "prod":
			classifications = []
			probs = self.class_prob()
			for classifier in self.classifiers:
				pred = classifier.predict(query)
				classes = pred[0]
				classifications.append(pred[1])
			pred = np.prod(classifications, axis=0) / list(probs.values())
		elif type == "maj":
			classifications = []
			for classifier in self.classifiers:
				pred = classifier.predict(query)
				classes = pred[0]
				classifications.append(pred[1])
			pred_table = np.zeros(len(classes))
			for prediction in classifications:
				proba = list(prediction)
				pred_table[proba.index(np.max(proba))] = pred_table[proba.index(np.max(proba))] + 1
			pred = pred_table / sum(pred_table)
		
		return classes, pred

	def score(self, type = self.type):
		count_right = 0
		count_peso = 0
		for q in range(len(self.X_test)):
			pred = self.predict(self.X_test[q], type=type)
			classes = list(pred[0])
			proba = list(pred[1])
			correto = self.y_test[q]
			if classes[proba.index(np.max(proba))] == correto:
				count_right = count_right + 1
		return count_right / len(self.X_test)

# Classe para definicoes das funcoes do modelo de classificacao Naive Bayes
class Naive:
	# funcao de inicializacao recebe os conjuntos de dados e o taxa de amostragem para teste (complementar a de treinamento)
	def __init__(self, X, y, test_size=0.1, alpha=1):
		self.X = X
		self.y = y
		self.test_size = test_size
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.train_test_split()
		self.alpha = alpha
		self.prob_class = self.class_prob()
		self.score = self.score()		

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
		count_right = 0
		count_peso = 0
		for q in range(len(self.X_test)):
			pred = self.predict(self.X_test[q])
			classes = list(pred[0])
			proba = list(pred[1])
			correto = self.y_test[q]
			if classes[proba.index(np.max(proba))] == correto:
				count_right = count_right + 1
				count_peso = count_peso + 1 / self.prob_class[correto]
		return [count_right / len(self.X_test), count_peso / len(self.X_test)]

#main
# Leitura do dataset e definicao dos conjuntos de dados X e y
data = pd.read_csv("data_discretizado.csv")
y = data["rating"]
X = data.drop(["Unnamed: 0","Unnamed: 0.1","rating","title"],axis=1) #remocao das colunas que nao serao validas na classificacao
score = 0
for i in range(10):
	e = Ensemble(X,y,n_classifiers=5)
	e.fit()
	score_aux = e.score()
	print(score_aux)
	score = score + score_aux
print("score medio: " + str(score/10))
