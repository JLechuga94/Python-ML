import pandas
import numpy as np
import matplotlib.pyplot as plt
# y = a + b * x
# X: 100 valores distribuidos segun una N(1.5, 2.5)
# Ye = 5 + 1.9 * x + e
# e estara distribuido segun una N(0, 0.8)
# Obtendremos los valores de beta y alfa para la regresión mejor ajustada a los datos
x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100)
y_pred = 5 + 1.9 * x
y_act = 5 + 1.9 * x + res

x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()

data = pandas.DataFrame(
    {
        "x": x_list,
        "y_actual": y_act_list,
        "y_prediccion": y_pred_list
    }
)

x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])
y_mean_pred = [np.mean(y_act) for i in range(1, len(x_list) + 1)]


data["beta_n"] = (data["x"] - x_mean)*(data["y_actual"] - y_mean) # Numerador
data["beta_d"] = (data["x"] - x_mean) ** 2 # Denominador
beta = sum(data["beta_n"])/sum(data["beta_d"])
alpha = y_mean - beta * x_mean

# y = a + b * x
data["y_model"] = alpha + beta * data["x"]

SSR = sum((data["y_model"]-y_mean) ** 2)
SSD = sum((data["y_model"]-data["y_actual"]) ** 2)
SST = sum((data["y_actual"]-y_mean) ** 2)
R2 = SSR/SST
print(data.head())
print(R2)

# RSE Residual Standard errores

RSE = np.sqrt(SSD/(len(data)-2))
print(RSE)
print(RSE/np.mean(data["y_actual"]))
# plt.figure(1)
# plt.plot(x, y_pred) # Modelo con valores seleccionados
# plt.plot(x, data["y_model"]) # Modelo con coeficientes calculados
# plt.plot(x, y_act, 'ro') # Datos reales del experimento
# plt.plot(x, y_mean_pred)
# plt.title("Valor real vs Modelo seleccionado vs Modelo B-A calculado")
# plt.show()
