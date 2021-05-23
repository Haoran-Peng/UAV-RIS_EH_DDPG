import numpy as np
import random as rd
import time
import math as mt
import sys
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random
import globe
import numpy as np
from gym.spaces.box import Box 

def _init(globe, readDistance = False, MaxStep = 600, filename=None):
	#the location of UAV-RIS
	globe.set_value('L_U', [80, 80, 20]) #[x, y, z]
	#the location of AP/BS
	globe.set_value('L_AP', [0, 0, 10])

	#CSI parameters
	globe.set_value('BW', mt.pow(10, 7)) #The bandwidth is 10 MHz
	# Noise power spectrum density is -174dBm/Hz;
	globe.set_value('N_0', mt.pow(10, ((-174 / 3) / 10)))
	globe.set_value('Xi', mt.pow(10, (3/10))) #the path loss at the reference distance D0 = 1m, 3dB;
	# urban env. from [Efficient 3-D Placement of an Aerial Base Station in Next Generation Cellular Networks] 
	#and [Joint Trajectory-Task-Cache Optimization with Phase-Shift Design of RIS-Assisted UAV for MEC]
	globe.set_value('a', 9.61)
	globe.set_value('b', 0.16)
	globe.set_value('eta_los', 1) 
	globe.set_value('eta_nlos', 20)

	# σ2 = -102dBm
	globe.set_value('AWGN', mt.pow(10, (-102/10)))
	# number of RIS antenna
	globe.set_value('N_ris', 100)
	#energy harvesting efficiency eta=0.7
	globe.set_value('eta', 0.7) 
	#path-loss exponent is α=3
	globe.set_value('alpha', 3)
	# additional attenuation factor φ is 20 dB
	globe.set_value('varphi', mt.pow(10, (20/10)))
	# max transmit Power from BS is 500W
	globe.set_value('P_max', 5 * mt.pow(10, 5))#mt.pow(10, 43/10)
	#number of user 
	globe.set_value('N_u', 1)
	# carrier frequency is 750 MHz
	globe.set_value('fc', 750 * mt.pow(10, 6))
	# speed of light
	globe.set_value('c', 3 * mt.pow(10, 8))
	#minimal requirement of sinr 12db
	globe.set_value('gamma_min', mt.pow(10, (12/10)))
	# the transmission power from the AP for a single user
	globe.set_value('power_i', 0.5 * mt.pow(10, 3))
	# the length of a time slot
	globe.set_value('t', int(MaxStep))
	#current time slot
	globe.set_value('step', 0)

	globe.set_value('kappa', mt.pow(10, (-30/10)))
	globe.set_value('hat_alpha', 2.5)
	if readDistance == True:
		p = filename
		with open(p, encoding = 'utf-8') as f:
			data = np.loadtxt(f, delimiter = ",")
			data.astype(np.int)
			globe.set_value('DistanceRU', data)
	else :
		globe.set_value('DistanceRU', D_RU(globe.get_value('t')))
	
	# tau is the radio of the energy-harvesting to time slot, default 0.5
	# lamda is the radio of the information transmit area, defalut 0.7

def _observation_space():
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    box = Box(low=20, high=60, shape=(1,),
               dtype=np.float32) 
    return box

def _action_space():
    # Return the actions space adjusted to match the shape of the processed
    # actions.
    box = Box(low=0, high=1, shape=(2,),
               dtype=np.float32) 
    return box

def pl_BR():
	L_U = globe.get_value('L_U')
	L_AP = globe.get_value('L_AP')
	a = globe.get_value('a')
	b = globe.get_value('b')
	varphi = globe.get_value('varphi')
	alpha = globe.get_value('alpha')

	theta = (180 / mt.pi) * mt.asin( ( (L_U[2] - L_AP[2]) / mt.sqrt(mt.pow(L_U[0], 2) + mt.pow(L_U[1], 2) + mt.pow((L_U[2] - L_AP[2]), 2))) )
	p_los = 1 + a * mt.exp(a * b - b * theta )
	p_los = 1 / p_los

	p_nlos = 1 - p_los
	# channel power gain (BS-RIS) with the los and nlos
	g_BR = (p_los + p_nlos * varphi) * mt.pow(mt.sqrt(mt.pow(L_U[0], 2) + mt.pow(L_U[1], 2) + mt.pow((L_U[2] - L_AP[2]), 2)), (0-alpha))
	
	return g_BR

def EH(tau, lamda):
	power_i = globe.get_value('power_i')
	eta = globe.get_value('eta')

	g_BR = pl_BR()
	power_total = power_i * globe.get_value('N_u')
	E_t = tau * eta * power_total * g_BR + (1 - tau) * (1 - lamda) * eta * power_total * g_BR
	return E_t

def D_RU(num):
	# d_ru = [];
	result = np.random.randint(20, 60, size=num)
	# print("====================")
	# print(result)
	# for i in range(0,num):
	# 	d_ru.append(random.randint(20,60))
	np.savetxt("distance.csv", result, delimiter=',')
	return result
	
def capacity (distance, tau, lamda):
	kappa = globe.get_value('kappa')
	hat_alpha = globe.get_value('hat_alpha')
	power_i = globe.get_value('power_i')
	AWGN = globe.get_value('AWGN')
	BW = globe.get_value('BW')

	for x in range(0,globe.get_value('N_u')):
		# d_ru = globe.get_value('DistanceRU')
		# print(step)
		# distance = d_ru[step]
		# print(distance)
		g_BR = pl_BR()
		signal =  power_i * g_BR * lamda * kappa * mt.pow((distance/1), -hat_alpha) * (1 - tau)
		interference = power_i * g_BR * lamda * kappa * mt.pow((distance/1), -hat_alpha) * (globe.get_value('N_u') - 1)
		if signal > 0:
			SINR = 10 * mt.log((signal/AWGN), 10)
		else:
			SINR = 0
		# print(kappa * mt.pow((20/1), -hat_alpha))
		# print(distance)
		# print(SINR)
		# tru = (1 - tau) * BW * mt.log((1+SINR), 2)
		# print(tru)
		return SINR

def env_state(step, tau, lamda):
	d_ru = globe.get_value('DistanceRU')
	if step < globe.get_value('t')-1:
		next_dru = d_ru[step+1]
	else:
		next_dru = d_ru[step]

	reward = EH(tau, lamda)
	SINR = capacity (d_ru[step], tau, lamda)
	if SINR < 12:
		reward = 0
	#radio_state = np.array([globe.get_value('power_i'), next_dru]) 
	radio_state = np.array([next_dru])
	# radio_state.append(globe.get_value('power_i'))
	# radio_state.append(next_dru)
	return reward, radio_state

def Step(a):
	t = globe.get_value('t')
	tau = a[0]
	lamda = a[1]
	step = globe.get_value('step')
	reward, radio_state = env_state(step, tau, lamda)
	done = False
	if step == t - 1:
		done = True

	globe.set_value('step', int(step+1))
	return radio_state, reward, done

def reset():
	globe.set_value('step', 0)
	d_ru = globe.get_value('DistanceRU')
	next_dru = d_ru[0]
	radio_state = np.array([next_dru])#np.array([globe.get_value('power_i'), next_dru])
	return radio_state	

def Grid_step(action, step):
	t = globe.get_value('t')
	tau = action[0]
	lamda = action[1]
	reward, radio_state = env_state(step, tau, lamda)
	done = False
	if step == t - 1:
		done = True

	return reward, done

def grid_reset(filename):
	globe.set_value('step', 0)

	p = filename
	with open(p, encoding = 'utf-8') as f:
		data = np.loadtxt(f, delimiter = ",")
		data.astype(np.int)
		globe.set_value('DistanceRU', data)

	d_ru = globe.get_value('DistanceRU')
	next_dru = d_ru[0]
	radio_state = np.array([next_dru])#np.array([globe.get_value('power_i'), next_dru])
	return radio_state

def reloadData(filename):
	with open(filename, encoding = 'utf-8') as f:
		data = np.loadtxt(f, delimiter = ",")
		data.astype(np.int)
		globe.set_value('DistanceRU', data)

