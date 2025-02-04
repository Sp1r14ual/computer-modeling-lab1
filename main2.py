import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.signal import ellip, filtfilt
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('data.csv', parse_dates=['DATE'], dayfirst=True)

# Преобразование курса в числовой формат
df['rate'] = pd.to_numeric(
    df['Euro/Czech koruna (EXR.D.CZK.EUR.SP00.A)'], errors='coerce')

# Удаление пропущенных значений
df.dropna(inplace=True)

# Сохранение оригинальных данных
df['original_rate'] = df['rate'].copy()

# Функция для сохранения графиков


def save_single_plot(data, filename, title, color='blue'):
    plt.figure(figsize=(12, 6))
    plt.plot(df['DATE'], data, color=color, linewidth=1.5)
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Курс')
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


# 1. Сохранение исходных данных
save_single_plot(df['original_rate'],
                 '01_original_data.png', 'Исходные данные')

# Этап 1: Обработка выбросов DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(df[['rate']])

# Восстановление выбросов
outliers = df['cluster'] == -1
df.loc[outliers, 'rate'] = df['rate'].rolling(window=3, min_periods=1).mean()

# 2. Данные после обработки выбросов
save_single_plot(df['rate'], '02_after_outliers.png',
                 'После обработки выбросов', 'green')

# Этап 2: Фильтрация шума
# Тест Бокса-Пирса
ljung_box_result = acorr_ljungbox(df['rate'], lags=[10], return_df=True)

# Применение эллиптического фильтра
b, a = ellip(5, 0.01, 120, 0.1, btype='low', analog=False)
df['filtered_rate'] = filtfilt(b, a, df['rate'])

# 3. Данные после фильтрации
save_single_plot(df['filtered_rate'], '03_after_filtering.png',
                 'После фильтрации', 'purple')

# Этап 3: Сглаживание
df['smoothed_rate'] = df['filtered_rate'].rolling(
    window=5, min_periods=1).mean()

# 4. Данные после сглаживания
save_single_plot(df['smoothed_rate'], '04_after_smoothing.png',
                 'После сглаживания', 'orange')

# Этап 4: Проверка стационарности
# ADF-тесты
adf_before = adfuller(df['original_rate'])
adf_after = adfuller(df['smoothed_rate'])

# KPSS-тесты
kpss_before = kpss(df['original_rate'], regression='c')
kpss_after = kpss(df['smoothed_rate'], regression='c')

# Вывод результатов тестов
print("\nРезультаты тестов стационарности:")
print(f"ADF тест (ориг. данные): p-value = {adf_before[1]:.4f}")
print(f"ADF тест (после обработки): p-value = {adf_after[1]:.4f}")
print(f"KPSS тест (ориг. данные): p-value = {kpss_before[1]:.4f}")
print(f"KPSS тест (после обработки): p-value = {kpss_after[1]:.4f}")

# Этап 5: Анализ тренда
df['diff'] = df['smoothed_rate'].diff()
trend_score = np.sign(df['diff']).sum()

# Интерпретация тренда
print("\nАнализ тренда Фостера-Стьюарта:")
if trend_score > 10:
    print("Обнаружен сильный восходящий тренд")
elif trend_score > 0:
    print("Обнаружен слабый восходящий тренд")
elif trend_score < -10:
    print("Обнаружен сильный нисходящий тренд")
elif trend_score < 0:
    print("Обнаружен слабый нисходящий тренд")
else:
    print("Значимый тренд не обнаружен")

# Вывод первых 5 строк обработанных данных
print("\nПервые 5 строк обработанных данных:")
print(df[['DATE', 'original_rate', 'rate',
          'filtered_rate', 'smoothed_rate']].head())
