import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

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
        return abs(1 / tE) * sigma_l + abs(-l / tE ** 2) * sigma_tiempo

    def sigma_vG(self, a, b, sigma_l):
        sigma_tiempo = 2  # del intervalo de tiempo
        tG = self.data.get_tG(a, b)
        l = self.data.get_divisiones(a, b)
        return abs(1 / tG) * sigma_l + abs(-l / tG ** 2) * sigma_tiempo

    def desviacion_ponderada(self, valores, errores):
        pesos = 1 / errores ** 2
        promedio_ponderado = np.average(valores, weights=pesos)
        varianza_ponderada = np.sum(pesos * (valores - promedio_ponderado) ** 2) / np.sum(pesos)
        return np.sqrt(varianza_ponderada)

    def carga_individual(self, a, b):  # También calcula su error
        dV = 0

        vE = self.data.get_vE(a, b)
        sigma_vE = self.sigma_vE(a, b, self.sigma_l(a, b))

        vG = self.data.get_vG(a, b)
        sigma_vG = self.sigma_vG(a, b, self.sigma_l(a, b))

        try:
            vE_avg = np.average(vE, weights=1 / sigma_vE) / 1000
            vG_avg = np.average(vG, weights=1 / sigma_vG) / 1000
        except ZeroDivisionError:
            print(f"[ERROR] División por cero al calcular promedios en rango ({a}, {b})")
            return np.nan, np.nan

        if vG_avg <= 0:
            print(f"[OMITIDO] vG_avg inválido (≤ 0) en el rango ({a}, {b}): {vG_avg:.4e} m/s")
            return np.nan, np.nan

        try:
            sigma_vG_avg = self.desviacion_ponderada(vG, sigma_vG) / 1000
            sigma_vE_avg = self.desviacion_ponderada(vE, sigma_vE) / 1000
        except Exception as e:
            print(f"[ERROR] Fallo al calcular desviación ponderada en rango ({a}, {b}): {e}")
            return np.nan, np.nan

        C = 6 * np.pi * d * np.sqrt((9 * eta ** 3) / (2 * g * rho))

        try:
            carga = (C / V) * (vG_avg + vE_avg) * np.sqrt(vG_avg)
        except Exception as e:
            print(f"[ERROR] Fallo al calcular carga en rango ({a}, {b}): {e}")
            return np.nan, np.nan

        # ---------- Cálculo de error -------------#
        dq_dV = -C * (vG_avg + vE_avg) * np.sqrt(vG_avg) / V ** 2
        dq_dvg = (C / V) * (np.sqrt(vG_avg) + (vG_avg + vE_avg) / (2 * np.sqrt(vG_avg)))
        dq_dvE = (C / V) * np.sqrt(vG_avg)

        sigma_q = np.sqrt((dq_dV * dV) ** 2 + (dq_dvg * sigma_vG_avg) ** 2 + (dq_dvE * sigma_vE_avg) ** 2)

        return carga, sigma_q

    def calcular_cargas_errores(self, rangos):
        cargas = []
        cargas_errores = []
        for inicio, fin in rangos:
            carga, error = self.carga_individual(inicio, fin)
            if not np.isnan(carga) and not np.isnan(error):
                cargas.append(carga)
                cargas_errores.append(error)
            else:
                print(f"[INFO] Rango ({inicio}, {fin}) descartado por datos inválidos.")



        return sorted(cargas), sorted(cargas_errores)


class MilikanPlotter:
    def __init__(self, cargas, errorCargas):
        self.cargas = cargas
        self.errorCargas = errorCargas

    def mostrar(self):
        # Configuración inicial
        fig, ax = plt.subplots(figsize=(6, 4))  # Tamaño un poco más grande

        # Personalización de ejes
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Configuración de la cuadrícula
        ax.grid(which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

        # Datos
        x = np.arange(len(self.cargas)) + 1

        # Gráfico de dispersión con mejor estilo
        scatter = ax.scatter(x, self.cargas,
                             c='#1f77b4',  # Color más atractivo
                             s=80,  # Tamaño de puntos
                             edgecolor='white',  # Borde blanco para mejor visibilidad
                             linewidth=0.7,
                             zorder=3)  # Para que aparezca sobre la cuadrícula

        # Barras de error con mejor estilo
        ax.errorbar(x, self.cargas, self.errorCargas,
                    fmt='none',
                    ecolor='#d62728',  # Color rojo distintivo
                    alpha=0.7,
                    capsize=4,  # Mayor tamaño de las caps
                    capthick=1.5,
                    elinewidth=1.5,
                    zorder=2)

        # Configuración de ticks
        plt.xticks(np.arange(min(x), max(x) + 1, 1))

        # Líneas de referencia mejoradas
        reference_colors = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#bcbd22']
        ref_values = [1.6e-19 * n for n in [1, 2, 3, 4, 6]]
        ranges = [(1, 2.5), (2.5, 3.5), (2.5, 3.5), (3.5, 4.5), (3.5, 4.5)]

        for val, (start, end), color in zip(ref_values, ranges, reference_colors):
            ax.hlines(val, start, end,
                      linestyles='dashed',
                      colors=color,
                      linewidth=1.5,
                      alpha=0.7,
                      label=f'{val:.1e} C')

        # Etiquetas y títulos
        ax.set_ylabel(r'Carga [C]', fontsize=14, labelpad=10)
        ax.set_xlabel(r'Número de medición', fontsize=14, labelpad=10)

        # Configuración de ticks
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Ajustar límites del eje y para mejor visualización
        y_min = min(self.cargas) - 0.2 * abs(min(self.cargas))
        y_max = max(self.cargas) + 0.2 * abs(max(self.cargas))
        ax.set_ylim(y_min, y_max)

        # Ajustes finales
        fig.tight_layout()

        # Mostrar con alta resolución
        plt.show(dpi=300)


if __name__ == "__main__":
    ruta_excel = r"F:\Facultad\Fisica Experimental\prog\Milikan\datosMilikan.xlsx"

    rangos = [(3, 9), (10, 20), (21, 30), (31, 39), (50, 55), (56, 63), (66, 75), (76, 81), (82, 86),
              (87, 95), (96, 105), (116, 125), (126, 134), (135, 144), (145, 150), (151, 164), (165, 169)]

    # (106, 115) valor demasiado alto
    # (40, 49) valor raro tambien

    data_loader = MilikanDataLoader(ruta_excel)
    calculator = MilikanCalculator(data_loader)
    cargas, errores = calculator.calcular_cargas_errores(rangos)

    print(cargas)

    plotter = MilikanPlotter(cargas, errores)
    plotter.mostrar()
