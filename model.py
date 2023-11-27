from pymongo import MongoClient
import pymongo
from pprint import pprint
import json
from bson.json_util import dumps

class DataModel:
    """Class for communicating with the non-relational database where
        the retailers' mined data is stored"""
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/", tls=False)
        self.database = self.client.get_database("TCC")
        self.collection = self.database.get_collection("data_sensors")

    def insert_document(self, json):
        self.collection.insert_one(json)

    def find_documents(self):
        db_mongo = self.collection.find()
        data = json.loads(dumps(list(db_mongo)))
        return data

    def find_documents_train(self):
        db_mongo_train = self.collection.find({"machine_state": "train"})
        data_train = json.loads(dumps(list(db_mongo_train)))

        return data_train

    def find_document_predict(self):
        # Agregação para obter os 25 últimos documentos para cada chave "sensor"
        pipeline = [
            {"$sort": {"created_at": -1}},  # Ordenar por created_at em ordem decrescente
            {"$group": {"_id": "$sensor", "documents": {"$push": "$$ROOT"}}},  # Agrupar por sensor e armazenar documentos
            {"$project": {"documents": {"$slice": ["$documents", 25]}}}  # Selecionar os últimos 25 documentos de cada grupo
        ]

        resultado_agregacao = list(self.collection.aggregate(pipeline))

        # Converter o resultado para um formato mais amigável (lista de documentos)
        documentos_finais = [documento for grupo in resultado_agregacao for documento in grupo['documents']]

        return documentos_finais

