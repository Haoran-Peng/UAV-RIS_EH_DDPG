import matplotlib.pyplot as plt
import sys
import csv
import numpy as np
import pandas as pd
import ARIS_ENV as arenv
import globe
from matplotlib import rcParams

##########################################################################
globe._init()
arenv._init(globe, True, 600, "Dataset/Distance_0.csv")
total_energy = arenv.EH(1,0) * 600
print(total_energy)

def get_DDPG_results(idx=20):
	total = []
	for i in range(idx):
		data = pd.read_csv('DDPG_Result/Dataset_'+str(i)+'_Rewards.csv',header=None)
		total.append(data[-1:])

	return total


def get_Exhaustive_results(idx=20):
	total = []
	for i in range(idx):
		data = pd.read_csv('Exhaustive_Result/Dataset_'+str(i)+'_rewards.csv', header=None, index_col=False)
		data = np.array(data, dtype=np.float)
		data = np.sum(data)
		total.append(data)

	return total

def get_A2C_results(idx=20):
	total = []
	for i in range(idx):
		data = pd.read_csv('A2C_Result/Dataset_'+str(i)+'_Rewards.csv',header=None)
		total.append(data[-1:])

	return total
	

def main(idx=20):
	DDPG_results = get_DDPG_results(idx)
	Exhaustive_results = get_Exhaustive_results(idx)
	A2C_results = get_A2C_results(idx)

	DDPG_results = np.array(DDPG_results, dtype=np.float)/total_energy
	Exhaustive_results = np.array(Exhaustive_results, dtype=np.float)/total_energy
	A2C_results = np.array(A2C_results, dtype=np.float)/total_energy

	print("DDPG_results mean: %.6f" % np.mean(DDPG_results))
	print("Exhaustive_results mean: %.6f" % np.mean(Exhaustive_results))
	print("A2C_results mean: %.6f" % np.mean(A2C_results))

	font = {'family': 'Times New Roman',
         'style': 'normal',
         'weight': 'bold',
        'size': 60,
        }

	x = np.linspace(1, idx, num=idx, endpoint=True, dtype=np.int)
	rcParams.update({'font.size': 60, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

	plt.plot(x, DDPG_results, ls='--', marker="o", color="coral", label="DDPG-based scheme", lw=4, markersize=20)
	plt.plot(x, Exhaustive_results, ls='--', marker="X", color="blue", label="Exhaustive search", lw=4, markersize=20)
	plt.plot(x, A2C_results, ls='-', marker="+", color="crimson", label="A2C-based scheme", lw=4, markersize=20)


	plt.xticks((2, 4, 6, 8, 10, 12, 14, 16, 18, 20), fontsize = 60)
	plt.yticks(fontsize = 60)

	plt.xlabel('Number of experiments', font = font)
	plt.ylabel('The ratio of the harvested energy\n to the received energy', font = font)

	plt.legend(loc='best', fontsize = 60)

	plt.show()

if __name__ == '__main__':
	idx = 20

	if len(sys.argv) > 1:
		idx = int(sys.argv[1])

	main(idx)