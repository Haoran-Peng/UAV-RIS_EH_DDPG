import numpy as np
import ARIS_ENV as arenv
import globe
import matplotlib.pyplot as plt

def plot(x, frame_idx, rewards):
    plt.figure()
    plt.title('Dataset_%i, frame %s. reward: %s' % (x, frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    plt.savefig('Exhaustive_Result/Dataset_'+str(x)+'.png', format='png')
    plt.close()

##########################################################################
maxStep = 600

globe._init()
arenv._init(globe, True, maxStep, "Dataset/Distance_1.csv")


lamda = np.linspace(0, 1, num=101, endpoint=True, dtype=np.float32)
tau = np.linspace(0, 1, num=101, endpoint=True, dtype=np.float32)

total_record = []
tauu=0
lamdaa = 0
def main():
	for x in range(20):
		print(x)
		rewards = []
		file = r"Dataset/Distance_"+str(x)+".csv"
		arenv.grid_reset(file)
		for step in range(maxStep):
			max_reward = 0
			for i in range(len(lamda)):
				for j in range(len(tau)):
					a = [tau[j], lamda[i]]
					r, done = arenv.Grid_step(a, step)
					if r > max_reward:
						max_reward = r
						tauu = tau[j]
						lamdaa = lamda[i]

			rewards.append(max_reward)
			# print("Step: ", step, "max_reward: %.6f" % max_reward)
			# print("tau: %.6f, lamda: %.6f" % (tauu, lamdaa))
			if step == maxStep - 1:
				np.savetxt("Exhaustive_Result/Dataset_"+str(x)+"_rewards.csv", rewards, delimiter=',')

		total_record.append(np.sum(rewards))

		plot(x, len(lamda)*len(tau)*step, rewards)

	np.savetxt("Exhaustive_Result/total_rewards.csv", total_record, delimiter=',')

if __name__ == '__main__':
	main()