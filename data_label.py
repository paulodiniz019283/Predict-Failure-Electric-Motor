import pandas as pd
from model import DataModel

docs = DataModel().find_documents()

dataframe_train = []

for doc in docs:
    dataframe_train.append({
        # "created_at": doc["created_at"],
        "sensor": doc["sensor"],
        "value": doc["value"],
        # "machine_state": doc["machine_state"],
        # "topic": doc["topic"]
    }
    )
dataframe_train = pd.DataFrame(dataframe_train)
dataframe_train = dataframe_train[dataframe_train['sensor'] != 'temperature']
dataframe_train['state'] = 'desalinhamento_mancais_eixo_motor'
dataframe_train.to_csv("dados_motor_estado_desalinhamento_mancais_eixo_motor.csv")
print(dataframe_train.head(30))

