from keras.models import load_model
import numpy as np
from model import DataModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class PredictFailure:
    def __init__(self):
        self.data_predict_failure = DataModel().find_document_predict()

    def load_and_predict_model(self, input_data, model_path='best_models/modelo_rnn_25_amostras.h5'):
        loaded_model = load_model(model_path)

        input_array = np.array(input_data).reshape(1, -1)

        prediction = loaded_model.predict(input_array)

        # Imprimir as porcentagens de possibilidade para ambas as classes
        probability_normal = prediction[0, 0] * 100
        probability_imbalance = (1 - prediction[0, 0]) * 100


        # Ajustar a previsão para 0 ou 1 (considerando um limiar de 0.5)
        prediction_binary = (prediction > 0.5).astype(int)

        return prediction_binary[0, 0], probability_normal, probability_imbalance

    def data_for_predict(self):
        dataframe_predict = []

        for doc in self.data_predict_failure:
            dataframe_predict.append({
                "sensor": doc["sensor"],
                "value": doc["value"],
            }
        )
        dataframe_predict = pd.DataFrame(dataframe_predict)
        columns_to_drop = ["accelerometer_y", "accelerometer_z", "accelerometer_x", "temperature"]
        dataframe_predict = dataframe_predict[~dataframe_predict.isin(columns_to_drop).any(axis=1)]
        dataframe_predict = self.enconder_data(dataframe_predict)
        list_data_predict = self.list_data_predict(dataframe_predict)

        return list_data_predict

    def enconder_data(self, dataframe_train):
        label_encoder = LabelEncoder()
        dataframe_train['sensor'] = label_encoder.fit_transform(
            dataframe_train['sensor']
        )

        # Criar um dicionário que mapeia rótulos para valores
        mapeamento_rotulos_valores = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

        # Imprimir o mapeamento
        for rotulo, valor in mapeamento_rotulos_valores.items():
            print(f"Rótulo: {rotulo}, Valor: {valor}")

        return dataframe_train

    def list_data_predict(self, df_for_group):

        list_data_predict = []
        list_sensors = df_for_group.sensor.unique()
        for sensor in list_sensors:
            list_data_predict_for_sensor = []
            df_for_group_filter = df_for_group[df_for_group['sensor'] == sensor]
            min_value = df_for_group_filter['value'].min()
            max_value = df_for_group_filter['value'].max()
            value = float(max_value) - float(min_value)

            list_data_predict_for_sensor.append(df_for_group_filter.sensor.unique()[0])
            list_data_predict_for_sensor.append(value)
            list_data_predict_for_sensor.append(float(min_value))
            list_data_predict_for_sensor.append(float(max_value))
            list_data_predict.append(list_data_predict_for_sensor)

        return list_data_predict


predict_failure = PredictFailure()
list_datas_predicts = predict_failure.data_for_predict()

for data_predict in list_datas_predicts:
    print(data_predict)
    result, porc_normal, porc_failure = predict_failure.load_and_predict_model(data_predict)
    print(f'A resposta do modelo é: {result}')
    print(f'Chance de ser normal: {porc_normal}')
    print(f'Chance de ser falha: {porc_failure}')
