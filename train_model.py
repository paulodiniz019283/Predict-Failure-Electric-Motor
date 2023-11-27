from model import DataModel
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class CreateModel:
    def __init__(self):
        self.docs_train = DataModel().find_documents_train()

    def run(self):
        # data_train = self.data_for_training()
        data_train = self.data_for_training_csv()
        data_train = self.data_processing_data_train(data_train)


        data_train = self.enconder_data(data_train)
        # data_train = self.normalize_data(data_train)

        X_train, X_test, y_train, y_test = self.separate_training_testing_validation(data_train)

        self.model_rnn(X_train, X_test, y_train, y_test)
        # self.model_cnn(X_train, X_test, y_train, y_test)
        # self.model_regre_log(X_train, X_test,  y_train, y_test)
        # self.model_random_florest(X_train, X_test,  y_train, y_test)
        # self.model_knn(X_train, X_test,  y_train, y_test)
        # self.model_naive_bayes(X_train, X_test,  y_train, y_test)
        # self.model_decision_tree(X_train, X_test,  y_train, y_test)
        # self.model_svm(X_train, X_test,  y_train, y_test)

    def data_for_training_csv(self):
        dataframe_train = pd.read_csv("dados/data_processing/df_completo_todos_dados_quantidade_amostras_25.csv")
        dataframe_train = dataframe_train[
            (dataframe_train['State'] == 'desbalanceamento') | (dataframe_train['State'] == 'normal')]
        return dataframe_train

    def data_for_training(self):
        dataframe_train = []

        for doc in self.docs_train:
            dataframe_train.append({
                # "created_at": doc["created_at"],
                "sensor": doc["sensor"],
                "value": doc["value"],
                # "machine_state": doc["machine_state"],
                # "topic": doc["topic"]
            }
        )

        dataframe_train = pd.DataFrame(dataframe_train)

        return dataframe_train

    def data_processing_data_train(self, dataframe_train):
        # dataframe_train = dataframe_train.head(50)
        # list = ['desalinhamento_mancais_eixo_motor-peso', 'normal', 'desalinhamento_mancais_base_motor', 'desbalanceamento']
        #
        # size_list = dataframe_train.shape[0]
        # result = [list[i % len(list)] for i in range(size_list)]
        # dataframe_train['State'] = result
        # dataframe_train = dataframe_train.drop(columns=["Min_Value", "Max_Value"])
        dataframe_train.rename(columns={"Subtraction": "Value"}, inplace=True)

        print(dataframe_train.columns)
        print(dataframe_train.State.unique())
        print(dataframe_train.State.value_counts())

        return dataframe_train

    def enconder_data(self, dataframe_train):
        label_encoder = LabelEncoder()
        dataframe_train['State'] = label_encoder.fit_transform(
            dataframe_train['State']
        )

        dataframe_train['Sensor'] = label_encoder.fit_transform(
            dataframe_train['Sensor']
        )

        # Criar um dicionário que mapeia rótulos para valores
        mapeamento_rotulos_valores = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Imprimir o mapeamento
        for rotulo, valor in mapeamento_rotulos_valores.items():
            print(f"Rótulo: {rotulo}, Valor: {valor}")

        return dataframe_train


    def normalize_data(self, dataframe_train):
        scaler = MinMaxScaler()
        dataframe_train['Value'] = scaler.fit_transform(
            dataframe_train[['Value']]
        )
        return dataframe_train

    def separate_training_testing_validation(self, dataframe_train):
        print(dataframe_train.columns)
        X_train = dataframe_train[['Sensor', 'Value', 'Min_Value', 'Max_Value']]
        # X_train = dataframe_train[['Sensor', 'Value']]

        y_train = dataframe_train['State']

        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_test,  y_train, y_test

    #Alterar
    def model_rnn(self, X_train, X_test, y_train, y_test):
        # Crie o modelo da rede neural
        model = keras.Sequential()

        # Camada de entrada
        model.add(layers.Input(shape=(X_train.shape[1],)))

        # Adicione 18 camadas intermediárias
        for _ in range(18):
            model.add(layers.Dense(units=64, activation='relu'))

        # Camada de saída
        model.add(layers.Dense(units=1, activation='sigmoid'))

        # Compile o modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Treine o modelo (substitua X_train e y_train pelos seus dados de treinamento reais)
        model.fit(X_train, y_train, epochs=13, batch_size=32)

        # Faça previsões no conjunto de teste
        y_pred = model.predict(X_test)

        # Converta para rótulos binários
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Exiba as métricas de desempenho
        accuracy = accuracy_score(y_test, y_pred_binary)
        confusion = confusion_matrix(y_test, y_pred_binary)
        print("Matriz de Confusão:")
        print(confusion)
        print(f'Acurácia do modelo: {accuracy}')

        report = classification_report(y_test, y_pred_binary)
        print("Relatório de Classificação:")
        print(report)

        # Plotar a matriz de confusão
        labels = sorted(set(y_test))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Rótulo Predito')
        plt.ylabel('Rótulo Real')
        plt.title('Matriz de Confusão')
        plt.show()

        model.save('best_models/modelo_rnn_25_amostras.h5')
        print(f'Modelo salvo em: best_models/modelo_rnn_25_amostras.h5')

    def model_random_florest(self, X_train, X_test, y_train, y_test):

        model = RandomForestClassifier(n_estimators=100,
                                       # criterion='entropy',
                                       random_state=42)  # Você pode ajustar o número de estimadores (árvores) conforme necessário
        model.fit(X_train, y_train)

        # Faça previsões no conjunto de teste
        y_pred = model.predict(X_test)

        # Avalie o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Exiba as métricas de desempenho
        print(f"Acurácia: {accuracy}")
        print("Matriz de Confusão:")
        print(confusion)
        print("Relatório de Classificação:")
        print(report)

    def model_cnn(self, X_train, X_test, y_train, y_test):
        print("Dimensões originais de X_train:", X_train.shape)

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        print("Dimensões de X_train após conversão para NumPy:", X_train.shape)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

        # Crie o modelo CNN
        model = models.Sequential()
        model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
        model.add(layers.MaxPooling2D((1, 1)))
        model.add(layers.Conv2D(64, (1, 1), activation='relu'))
        model.add(layers.MaxPooling2D((1, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))  # Para classificação binária, use sigmoid

        # Compile o modelo
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',  # Para classificação binária
                      metrics=['accuracy'])

        # Treine o modelo
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

        # Avalie o desempenho do modelo
        y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Converte probabilidades em classes binárias

        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Exiba as métricas de desempenho
        print(f"Acurácia: {accuracy}")
        print("Matriz de Confusão:")
        print(confusion)
        print("Relatório de Classificação:")
        print(report)

    def model_regre_log(self, X_train, X_test, y_train, y_test):
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Faça previsões no conjunto de teste
        y_pred = model.predict(X_test)

        # Avalie o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Exiba as métricas de desempenho
        print(f"Acurácia: {accuracy}")
        print("Matriz de Confusão:")
        print(confusion)
        print("Relatório de Classificação:")
        print(report)

    #ACABA
    def model_naive_bayes(self, X_train, X_test, y_train, y_test):
        y_pred = self.train_model_naive_bayes(X_train, X_test, y_train)
        self.metrics_train_model_naive_bayes(y_pred, y_test)

    @staticmethod
    def train_model_naive_bayes(X_train, X_test, y_train):
        naive_bayes_model = GaussianNB()
        naive_bayes_model.fit(X_train, y_train)

        y_pred = naive_bayes_model.predict(X_test)
        return y_pred

    @staticmethod
    def metrics_train_model_naive_bayes(y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusão Naive Bayes:")
        print(confusion)
        print(f'Acurácia do modelo Naive Bayes: {accuracy}')

        report = classification_report(y_test, y_pred)
        print("Relatório de Classificação:")
        print(report)

        # Plotar a matriz de confusão
        labels = sorted(set(y_test))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel('Rótulo Predito')
        plt.ylabel('Rótulo Real')
        plt.title('Matriz de Confusão - Naive Bayes')
        plt.show()

    def model_decision_tree(self, X_train, X_test, y_train, y_test):
        y_pred = self.train_model_naive_bayes(X_train, X_test, y_train)
        self.metrics_train_model_decision_tree(y_pred, y_test)

    @staticmethod
    def train_model_decision_tree(X_train, X_test, y_train):
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        decision_tree_model.fit(X_train, y_train)

        y_pred = decision_tree_model.predict(X_test)

        return y_pred

    @staticmethod
    def metrics_train_model_decision_tree(y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusão Árvore de Decisão:")
        print(confusion)
        print(f'Acurácia do modelo de Árvore de Decisão: {accuracy}')
        report = classification_report(y_test, y_pred)
        print("Relatório de Classificação:")
        print(report)

    def model_knn(self, X_train, X_test, y_train, y_test):
        model = KNeighborsClassifier(n_neighbors=25)  # Você pode ajustar o número de vizinhos (k) conforme necessário
        model.fit(X_train, y_train)

        # Faça previsões no conjunto de teste
        y_pred = model.predict(X_test)

        # Avalie o desempenho do modelo
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Exiba as métricas de desempenho
        print(f"Acurácia: {accuracy}")
        print("Matriz de Confusão:")
        print(confusion)
        print("Relatório de Classificação:")
        print(report)

    def model_svm(self, X_train, X_test, y_train, y_test):
        y_pred = self.train_model_svm(X_train, X_test, y_train)
        self.metrics_train_model_svm(y_pred, y_test)

    @staticmethod
    def train_model_svm(X_train, X_test, y_train):
        svm_model = SVC(kernel='linear', random_state=42)
        svm_model.fit(X_train, y_train)

        y_pred = svm_model.predict(X_test)

        return y_pred

    @staticmethod
    def metrics_train_model_svm(y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        print("Matriz de Confusão SVM:")
        print(confusion)
        print(f'Acurácia do modelo de SVM: {accuracy}')
        report = classification_report(y_test, y_pred)
        print("Relatório de Classificação:")
        print(report)

create_model = CreateModel()
create_model.run()
