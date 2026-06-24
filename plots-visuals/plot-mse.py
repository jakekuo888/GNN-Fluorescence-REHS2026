import os
import matplotlib.pyplot as plt

#Plotting MSE

train_MSE = []
test_MSE = []

with open("./data/plot-data/MSE.txt", "r") as MSE:
	MSE_Data = MSE.read().splitlines()
	
	for i in MSE_Data:
		train_MSE.append(float(i.split(',')[0]))
		test_MSE.append(float(i.split(',')[1]))

plt.plot(train_MSE, color = 'blue', linestyle = '-', label = 'train')
plt.plot(test_MSE, color = 'red', linestyle = '--', label = 'test')
plt.legend()
plt.title(f'Absorption MAE (nm) vs Epoch ({len(train_MSE)})')
plt.xlabel('Epoch')
plt.ylabel('Absorption MAE (nm)')

plt.savefig('./plots-visuals/new-plots/NM-Difference.png')