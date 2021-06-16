import numpy as np
################################
#excuse this file will generate 20 distance dataset.
################################

def D_RU(num, filename):
	result = np.random.randint(20, 60, size=num)
	np.savetxt("Dataset/" + filename + ".csv", result, delimiter=',')
	return result

def main():
	for x in range(20):
		D_RU(600, "Distance_"+str(x))

if __name__ == '__main__':
	main()
	