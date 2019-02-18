import numpy as np
import torch
import json
from azureml.core.model import Model
import model

def init():
    try:
        global modelo
        modelPath = Model.get_model_path("miRed")

        modelo = torch.load(modelPath)
        modelo.eval()
    except Exception as e:
        result = str(e)
        return json.dumps({'error': result})


def run(jsonData):
    try:
        data = np.asarray(json.loads(jsonData)['data'])
        data = data.reshape(1, 2)
        data = torch.from_numpy(data).requires_grad_(True).float()

        out = modelo(data)

        _, salida = torch.max(out, 1)
        salida = salida.numpy().tolist()
        return json.dumps({'resultado': salida})

    except Exception as e:
        result = str(e)
        return json.dumps({'error': result})
