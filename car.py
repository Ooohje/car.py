import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일이 있는 디렉토리 경로
data_dir = "C:/Users/leejungmin/Desktop/caracel"

# 신호크기 범위(SMA) 계산 함수
def calculate_sma(data):
    return np.mean(np.abs(data))

# 신호 벡터 크기(SVM) 계산 함수
def calculate_svm(data):
    return np.sqrt(np.sum(data ** 2))

total_sma = 0
total_svm = 0
num_files = 0

sma_values = []
svm_values = []

# 모든 CSV 파일을 처리
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_dir, filename)
        
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 가속도 데이터 추출
        acceleration_data = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']]
        
        # 신호크기 범위(SMA) 계산
        sma = calculate_sma(acceleration_data)
        
        svm_values_file = []
        
        # 신호 벡터 크기(SVM) 계산
        for index, row in acceleration_data.iterrows():
            svm = calculate_svm(row.values)  # 각 행의 값을 Numpy 배열로 변환하여 계산
            svm_values_file.append(svm)
        
        average_svm = np.mean(svm_values_file)
        
        # 결과 출력
        print(f"File: {filename}, SMA: {sma}, SVM: {average_svm}")
        
        sma_values.append(sma)
        svm_values.append(average_svm)
        
        # 각 csv 파일 그래프 그리기
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], acceleration_data)
        plt.title(f"Acceleration Data - {filename}")
        plt.xlabel("Time")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend(['ax', 'ay', 'az'])
        plt.grid()
        plt.show()
        '''
        
avg_sma = np.mean(sma_values)
avg_svm = np.mean(svm_values)
std_sma = np.std(sma_values)
std_svm = np.std(svm_values)
print(f'aver_SMA: {avg_sma}, std_SMA: {std_sma}, aver_SVM: {avg_svm}, std_SVM: {std_svm}')


#표준편차 값이 그리기
plt.figure(figsize=(10, 6))
plt.plot(sma_values, label='SMA')
plt.axhline(y=avg_sma, color='r', linestyle='--', label='Average SMA')
plt.errorbar(range(len(sma_values)), sma_values, yerr=std_sma, fmt='o', color='r', alpha=0.5)
plt.title("SMA Values for Each CSV File")
plt.xlabel("File Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(svm_values, label='SVM')
plt.axhline(y=avg_svm, color='b', linestyle='--', label='Average SVM')
plt.errorbar(range(len(svm_values)), svm_values, yerr=std_svm, fmt='o', color='b', alpha=0.5)
plt.title("SVM Values for Each CSV File")
plt.xlabel("File Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sma_values, label='SMA')
plt.plot(svm_values, label='SVM')
plt.axhline(y=avg_sma, color='r', linestyle='--', label='Average SMA')
plt.axhline(y=avg_svm, color='b', linestyle='--', label='Average SVM')
plt.errorbar(range(len(sma_values)), sma_values, yerr=std_sma, fmt='o', color='r', alpha=0.5)
plt.errorbar(range(len(svm_values)), svm_values, yerr=std_svm, fmt='o', color='b', alpha=0.5)
plt.title("SMA and SVM Values for Each CSV File")
plt.xlabel("File Index")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()