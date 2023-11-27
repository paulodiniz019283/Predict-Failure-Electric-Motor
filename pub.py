import datetime
import random
from paho.mqtt import client as mqtt_client
from model import DataModel
import ast
import json

broker = 'broker.emqx.io'
port = 1883
topic_temperature = "data_sensor_temperature"
topic_gyro_x = "data_sensor_gyro_x"
topic_gyro_y = "data_sensor_gyro_y"
topic_gyro_z = "data_sensor_gyro_z"
topic_accelerometer_x = "data_sensor_accelerometer_x"
topic_accelerometer_y = "data_sensor_accelerometer_y"
topic_accelerometer_z = "data_sensor_accelerometer_z"

client_id = f'subscribe-{random.randint(0, 100)}'


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(msg.payload.decode())
        parts = msg.payload.decode().split(",")
        value = parts[0]
        sensor = parts[1]
        state = parts[2]
        data = {
            "sensor": sensor,
            "value": value,
            "topic": msg.topic,
            "machine_state": state,
            "created_at": datetime.datetime.now()
        }
        insert_data_sensor_db(data)
        print(data)

    client.subscribe(topic_temperature)
    client.subscribe(topic_gyro_x)
    client.subscribe(topic_gyro_y)
    client.subscribe(topic_gyro_z)
    client.subscribe(topic_accelerometer_x)
    client.subscribe(topic_accelerometer_y)
    client.subscribe(topic_accelerometer_z)

    client.on_message = on_message


def insert_data_sensor_db(data):
    DataModel().insert_document(data)

def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()
