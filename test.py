import numpy as np
import torch
import score
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


# global model

# model = torch.load("C:/Users/Diego/Documents/DeployAzure/Red.pt")
#
# model.eval()
#
# datos = np.random.rand(2).reshape(1, 2)
# print(datos)
#
# datos = torch.from_numpy(datos).requires_grad_(True).float()
#
# out = model(datos)
# _, salida = torch.max(out, 1)
# salida = salida.numpy()
#
# print(salida)

score.init()
numeros = np.array([0.25, 0.25])
envio = {'data': numeros}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

envioJson = json.dumps(envio, cls=NumpyEncoder)

print(score.run(envioJson))
