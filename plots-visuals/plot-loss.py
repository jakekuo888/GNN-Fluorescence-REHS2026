import os
import matplotlib.pyplot as plt

#Plotting Loss

train_loss = []
test_loss= []

with open("./data/plot-data/loss.txt", "r") as loss:
	MSE_Data = loss.read().splitlines()
	
	for i in MSE_Data:
		train_loss.append(float(i.split(',')[0]))
		test_loss.append(float(i.split(',')[1]))

plt.plot(train_loss, color = 'blue', linestyle = '-', label = 'train')
plt.plot(test_loss, color = 'red', linestyle = '--', label = 'test')
plt.legend()
plt.title(f'Absorption MAE (nm) vs Epoch ({len(train_loss)})')
plt.xlabel('Epoch')
plt.ylabel('Absorption MAE (nm)')

plt.savefig('./plots-visuals/new-plots/NM-Difference.png')