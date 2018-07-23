import pandas
import numpy as np
import matplotlib.pyplot as plt
# y = a + b * x
# X: 100 valores distribuidos segun una N(1.5, 2.5)
# Ye = 5 + 1.9 * x + e
# e estara distribuido segun una N(0, 0.8)

x = 1.5 + 2.5 * np.random.randn(100)
res = 0 + 0.8 * np.random.randn(100)
y_pred = 5 + 1.9 * x
y_act = 5 + 1.9 * x + res

x_list = x.tolist()
y_pred_list = y_pred.tolist()
y_act_list = y_act.tolist()

y_mean = [np.mean(y_act) for i in range(1, len(x_list) + 1)]
data = pandas.DataFrame(
    {
        "x": x_list,
        "y_actual": y_act_list,
        "y_prediccion": y_pred_list
    }
)

# Cálculo de las diferencias entre predicciones y observados
data["SSR"] = (data["y_prediccion"]-np.mean(y_act)) ** 2
data["SSD"] = (data["y_prediccion"]-data["y_actual"]) ** 2
data["SST"] = (data["y_actual"]-np.mean(y_act)) ** 2

# Suma de los errores entre la predicción y los datos observados
SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])
# Porcentaje de paridad entre la predicción y los valores observados
# Esto nos dice que tan buena aproximación fue la preddición
R2 = SSR/SST

print(data.head())
print(SSR)
print(SSD)
print(SST)
print(R2)
print(SSR+SSD)

plt.figure(1)
plt.plot(x, y_pred)
plt.plot(x, y_act, 'ro')
plt.plot(x, y_mean)
plt.title("Valor actual vs Predicción")
plt.figure(2)
plt.hist(data["y_prediccion"]-data["y_actual"])
plt.show()
