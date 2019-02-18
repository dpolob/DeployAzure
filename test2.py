import numpy as np
import json
import requests


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


numeros = np.array([0.5, 0.5])
envio = {'data': numeros}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

envioJson = json.dumps(envio, cls=NumpyEncoder)
r = requests.post("http://52.157.225.158:80/score", envioJson, headers=headers)
print(r.text)
