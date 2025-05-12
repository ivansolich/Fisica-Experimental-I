import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from sklearn.cluster import KMeans

g = 9.81  # m/s^2
eta = 1.8e-5  # N.s/m2
rho = 800  # kg/m^3
d = 0.004  # m
V = 318  # V

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


class MilikanDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_excel(self.filepath, sheet_name="datos")
        except Exception as e:
            print(f"Error al cargar el archivo Excel: {e}")
            exit()

    def get_divisiones(self, a, b):
        return self.df.iloc[a - 1:b, 1].values.astype(float)

    def get_tE(self, a, b):  # Intervalo de tiempo de velocidad subida
        return self.df.iloc[a - 1:b, 5].values.astype(float)

    def get_tG(self, a, b):
        return self.df.iloc[a - 1:b, 11].values.astype(float)

    def get_vE(self, a, b):
        return self.df.iloc[a - 1:b, 6].values.astype(float)

    def get_vG(self, a, b):
        return self.df.iloc[a - 1:b, 11].values.astype(float)


class MilikanCalculator:
    def __init__(self, data_loader):
        self.data = data_loader

    def sigma_l(self, a, b):
        sigma_division = 0.01  # cada división es 0.04 mm
        divisiones = self.data.get_divisiones(a, b)
        return divisiones * sigma_division

    def sigma_vE(self, a, b, sigma_l):
        sigma_tiempo = 2  # del intervalo de tiempo
        tE = self.data.get_tE(a, b)
        l = self.data.get_divisiones(a, b)
        return np.sqrt((1 / tE * sigma_l) ** 2 + (-l / tE ** 2 * sigma_tiempo) ** 2)

    def sigma_vG(self, a, b, sigma_l):
        sigma_tiempo = 2  # del intervalo de tiempo
        tG = self.data.get_tG(a, b)
        l = self.data.get_divisiones(a, b)
        return np.sqrt((1 / tG * sigma_l) ** 2 + (-l / tG ** 2 * sigma_tiempo) ** 2)

    def desviacion_ponderada(self, valores, errores):
        pesos = 1 / errores ** 2
        promedio_ponderado = np.average(valores, weights=pesos)
        varianza_ponderada = np.sum(pesos * (valores - promedio_ponderado) ** 2) / np.sum(pesos)
        return np.sqrt(varianza_ponderada)

    def carga_individual(self, a, b):  # Tambien calcula su error
        dV = 3

        vE = self.data.get_vE(a, b)
        sigma_vE = self.sigma_vE(a, b, self.sigma_l(a, b))

        vG = self.data.get_vG(a, b)
        sigma_vG = self.sigma_vG(a, b, self.sigma_l(a, b))

        vE_avg = np.average(vE, weights=1 / sigma_vE) / 1000
        vG_avg = np.average(vG, weights=1 / sigma_vG) / 1000

        sigma_vG_avg = self.desviacion_ponderada(vG, sigma_vG) / 1000
        sigma_vE_avg = self.desviacion_ponderada(vE, sigma_vE) / 1000

        C = 6 * np.pi * d * np.sqrt((9 * eta ** 3) / (2 * g * rho))

        carga = (C / V) * (vG_avg + vE_avg) * np.sqrt(vG_avg)

        #---------- CALCULO DE ERROR -------------#


        return carga, sigma_q

    def calcular_cargas_errores(self, rangos):
        cargas = []
        cargas_errores = []
        for inicio, fin in rangos:
            carga, error = self.carga_individual(inicio, fin)
            cargas.append(carga)
            cargas_errores.append(error)

        return sorted(cargas), sorted(cargas_errores)  # sorted() para que quede de min a max

    def carga_electron_representativa(self, rangos):
        cargas, e = self.calcular_cargas_errores(rangos)

        cargas = np.array(cargas)

        X = cargas.reshape(-1, 1)
        kmeans = KMeans(n_clusters=6).fit(X)
        centros = np.sort(kmeans.cluster_centers_.flatten())
        e_estimado = np.mean(np.diff(centros))

        sigma_e = np.std(np.diff(centros), ddof=1)  # ddof=1 porque es una muestra, no tocar

        print(f"e = {e_estimado:.3e} ± {sigma_e:.3e} C")

        print(f"Centros de clusters: {centros}")


class MilikanPlotter:
    def __init__(self, cargas, errorCargas):
        self.cargas = cargas
        self.errorCargas = errorCargas

    def mostrar(self):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        x = np.arange(len(self.cargas)) + 1
        ax.scatter(x, self.cargas, s=20, color='black')

        ax.errorbar(x, self.cargas, self.errorCargas, fmt='none', ecolor='#7f7f7f', alpha=0.5, capsize=2)

        plt.xticks(np.arange(min(x), max(x) + 1, 1))
        plt.hlines(1.6e-19, 1, 4, linestyles='-', linewidths=1)
        plt.hlines(1.6e-19 * 2, 5, 8, linewidths=1)
        plt.hlines(1.6e-19 * 3, 9, 14, linewidths=1)
        plt.hlines(1.6e-19 * 4, 15, 17, linewidths=1)
        plt.hlines(1.6e-19 * 5, 17.5, 18.5, linewidths=1)
        plt.hlines(1.6e-19 * 6, 19, 21, linewidths=1)

        ax.set_ylabel(r'Carga [C]', fontsize=18)
        ax.set_xlabel(r'Número de medición', fontsize=18)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        fig.tight_layout()
        plt.show(dpi=300)


if __name__ == "__main__":
    ruta_excel = r"F:\Facultad\Fisica Experimental\prog\Milikan\datosMilikan.xlsx"

    rangos = [(3, 9), (10, 20), (21, 30), (31, 39), (50, 55), (56, 63), (66, 75), (76, 81), (82, 86),
              (87, 95), (96, 105), (116, 125), (126, 134), (135, 144), (145, 150), (151, 164), (165, 169),
              (170, 175), (176, 185), (196, 205), (206, 211)]

    """rangos = [(3, 9), (10, 20), (21, 30), (31, 39), (50, 55), (56, 63), (66, 75), (76, 81), (82, 86),
              (87, 95), (96, 105), (116, 125), (126, 134), (135, 144), (145, 150), (151, 164), (165, 169),
              (170, 175), (176, 185), (196, 205), (206, 211), (40, 49), (106, 115), (186, 195)]"""

    # (186, 195) alto
    # (106, 115) valor demasiado alto
    # (40, 49) valor alto tambien

    data_loader = MilikanDataLoader(ruta_excel)
    calculator = MilikanCalculator(data_loader)
    cargas, errores = calculator.calcular_cargas_errores(rangos)

    calculator.carga_electron_representativa(rangos)

    plotter = MilikanPlotter(cargas, errores)
    plotter.mostrar()

from scipy.optimize import minimize_scalar

def error(e_candidato):
    n = cargas / e_candidato
    return np.sum((n - np.round(n)) ** 2)


resultado = minimize_scalar(error, bounds=(1.3e-19, 1.7e-19), method='bounded')

e_estimado = resultado.x
print(f"Valor estimado de e: {e_estimado:.4e} C")

e_vals = np.linspace(1.3e-19, 1.7e-19, 1000)
error_vals = [error(e) for e in e_vals]


def estimar_carga_electron_con_incertidumbre(cargas, max_n=20):
    cargas = np.asarray(cargas)
    q_r = cargas[0]

    def calcular_error(n):
        ratios = cargas / q_r
        productos = ratios * n
        diferencias = productos - np.round(productos)
        return np.sum(diferencias ** 2)

    ns = np.arange(1, max_n + 1)
    errores = np.array([calcular_error(n) for n in ns])

    n_r_optimo = ns[np.argmin(errores)]

    ratios = cargas / q_r
    n_i_enteros = np.round(ratios * n_r_optimo)

    mask = n_i_enteros != 0
    cargas_filtradas = cargas[mask]
    n_i_enteros_filtrados = n_i_enteros[mask]

    e_individuales = cargas_filtradas / n_i_enteros_filtrados

    e_estimado = np.mean(e_individuales)
    incertidumbre = np.std(e_individuales, ddof=1) / np.sqrt(len(e_individuales))  # Error estándar de la media

    return e_estimado, incertidumbre, n_r_optimo, errores, e_individuales


if __name__ == "__main__":
    cargas_medidas = cargas

    e_estimado, incertidumbre, n_optimo, errores, e_individuales = estimar_carga_electron_con_incertidumbre(
        cargas_medidas)

    print(f"Carga elemental estimada: ({e_estimado:.4e} ± {incertidumbre:.4e}) C")
    print(f"Valor óptimo de n_r: {n_optimo}")