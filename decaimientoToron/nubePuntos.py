import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

from matplotlib.ticker import AutoMinorLocator

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

excel_file = r"F:\Facultad\Fisica Experimental\Exp1.xlsx"

try:
    df = pd.read_excel(excel_file, sheet_name="datos1")

    columna_y = 'Unnamed: 5'  # Esto sería la columna F (índice 5)
    columna_x = 'Unnamed: 6'  # Esto sería la columna G (índice 6)

except Exception as e:
    print(f"Error al cargar el archivo Excel: {e}")
    exit()


def decayTime(a, b):
    serie = np.array([
        df.iloc[a - 1:b, 5].values.astype(float),  # Columna F (índice 5)
        df.iloc[a - 1:b, 6].values.astype(float)  # Columna G (índice 6) # Promedio
    ])

    reciproco = 1 / serie[0]
    log = np.log10(reciproco)
    err_y = np.abs(-1 / (serie[0] * np.log(10))) * 2
    weights = 1 / (err_y ** 2)
    fit = np.polyfit(serie[1], log, 1, cov=True, w=weights)

    ordenada = fit[0][1]

    logNormalizado = log - ordenada

    return serie[1], logNormalizado, err_y


def plotFit(log, promedio, fit, err_y):
    log = np.array(log)
    promedio = np.array(promedio)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.grid(which='major', linestyle='-', color='gray', alpha=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    t = np.linspace(min(promedio), max(promedio), 300)

    pendiente = fit[0][0]
    ordenada = fit[0][1]

    linea_ajuste = pendiente * t + ordenada

    predicciones = pendiente * promedio + ordenada
    residuos = log - predicciones

    n = len(promedio)
    s_residuos = np.sqrt(np.sum(residuos**2) / (n - 2))

    promedio_promedio = np.mean(promedio)
    suma_cuadrados = np.sum((promedio - promedio_promedio)**2)

    confianza = 0.997
    alpha = 1 - confianza
    t_critico = stats.t.ppf(1 - alpha/2, df=n-2)

    error_confianza = s_residuos * np.sqrt(1/n + (t - promedio_promedio)**2 / suma_cuadrados)

    error_prediccion = s_residuos * np.sqrt(1 + 1/n + (t - promedio_promedio)**2 / suma_cuadrados)

    limite_superior_confianza = linea_ajuste + t_critico * error_confianza
    limite_inferior_confianza = linea_ajuste - t_critico * error_confianza

    limite_superior_prediccion = linea_ajuste + t_critico * error_prediccion
    limite_inferior_prediccion = linea_ajuste - t_critico * error_prediccion

    ax.scatter(promedio, log, color='black', marker="o", s=15)
    ax.plot(t, linea_ajuste, color='#1f77b4', ls="--", alpha=1, label="Regresión Lineal")

    # Intervalo de confianza
    ax.fill_between(t, limite_inferior_confianza, limite_superior_confianza,
                    color='#1f77b4', alpha=0.3, label="Intervalo de Confianza")

    ax.fill_between(t, limite_inferior_prediccion, limite_superior_prediccion,
                    color='#ff7f0e', alpha=0.2, label="Intervalo de Predicción")

    ax.errorbar(promedio, log, yerr=err_y, fmt='none', ecolor='#7f7f7f', alpha=0.5, capsize=2)

    ax.set_xlabel('Tiempo [s]', fontsize=18)
    ax.set_ylabel('log(1/S) [1/s]', fontsize=18)
    plt.xticks(fontsize=12)  # Tamaño de números en eje X
    plt.yticks(fontsize=12)  # Tamaño de números en eje Y

    ax.legend()

    fig.tight_layout()
    plt.savefig('nubePuntos.png', dpi=300)
    plt.show(dpi=300)






rangos = [
    (18, 24), (26, 30),
    (33, 35), (38, 41), (43, 46), (49, 51), (54, 57),
    (62, 64), (66, 71), (73, 77), (80, 85), (88, 95)
]

logNormalizadoTotal = []
promedioTotal = []
linear_error = []

for i, (inicio, fin) in enumerate(rangos, 1):
    try:
        promedio, logNormalizado, err_y = decayTime(inicio, fin)
        promedioTotal.extend(promedio)  # Añade elementos del array uno por uno
        logNormalizadoTotal.extend(logNormalizado)
        linear_error.extend(err_y)


    except Exception as e:
        print(f"Error procesando serie {i}: {e}")

fit = np.polyfit(promedioTotal, logNormalizadoTotal, 1, cov=True)

time = -0.301 / fit[0][0]
errorTiempo = (0.301 / ((fit[0][0]) ** 2)) * (3 * np.sqrt(fit[1][0][0]))
print(f"Tiempo: {time:.1f}, Error: {errorTiempo:.1f}")

cte_desintegracion = np.log(2) / time
delta_lambda = (np.log(2) / time ** 2) * errorTiempo

print(f"Constante de desintegración: {cte_desintegracion*1000:.1f} \\ Error: {delta_lambda*1000:.1f}")

plotFit(logNormalizadoTotal, promedioTotal, fit, linear_error)