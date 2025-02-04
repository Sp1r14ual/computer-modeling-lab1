import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.signal import ellip, filtfilt

# Загрузка данных
df = pd.read_csv('data.csv', parse_dates=['DATE'], dayfirst=True)

# Преобразуем курс в тип float
df['rate'] = pd.to_numeric(
    df['Euro/Czech koruna (EXR.D.CZK.EUR.SP00.A)'], errors='coerce')

# Удалим строки с пропущенными значениями
df.dropna(inplace=True)

# Этап 1: Удаление выбросов с использованием DBSCAN
# Используем DBSCAN для обнаружения выбросов (настроим параметры на основе данных)
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(df[['rate']])

# Восстановим выбросы (кластеры с меткой -1) на основе соседних значений
outliers = df['cluster'] == -1
df.loc[outliers, 'rate'] = df['rate'].rolling(window=3, min_periods=1).mean()

# Этап 2: Фильтрация шума
# (a) Статистика Бокса-Пирса для теста на белый шум
ljung_box_result = acorr_ljungbox(df['rate'], lags=[10], return_df=True)
print(f"Статистика Бокса-Пирса:\n{ljung_box_result}")

# (b) Применение эллиптического фильтра
b, a = ellip(5, 0.01, 120, 0.1, btype='low', analog=False)
df['filtered_rate'] = filtfilt(b, a, df['rate'])

# Этап 3: Сглаживание с использованием взвешенного скользящего среднего
df['smoothed_rate'] = df['filtered_rate'].rolling(
    window=5, min_periods=1).mean()

# Этап 4: Проверка стационарности до и после обработки (ADF и KPSS)
# ADF-тест
adf_result_before = adfuller(df['rate'])
adf_result_after = adfuller(df['smoothed_rate'])

# KPSS-тест
kpss_result_before = kpss(df['rate'], regression='c')
kpss_result_after = kpss(df['smoothed_rate'], regression='c')

print(f"ADF-тест до обработки: p-значение = {adf_result_before[1]}")
print(f"ADF-тест после обработки: p-значение = {adf_result_after[1]}")
print(f"KPSS-тест до обработки: p-значение = {kpss_result_before[1]}")
print(f"KPSS-тест после обработки: p-значение = {kpss_result_after[1]}")

# Этап 5: Проверка на наличие тренда методом Фостера-Стьюарта
df['diff'] = df['smoothed_rate'].diff()
trend_foster_stewart = np.sign(df['diff']).sum()

if trend_foster_stewart > 0:
    print("В данных присутствует восходящий тренд.")
elif trend_foster_stewart < 0:
    print("В данных присутствует нисходящий тренд.")
else:
    print("Тренд не выявлен.")

# Выводим первые 5 строк для проверки результатов
print(df[['DATE', 'rate', 'filtered_rate', 'smoothed_rate']])
