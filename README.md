# Long-Lasting UAV-aided RIS Communications based on SWIPT
## Introduction
This repository is the implementation of "Long-Lasting UAV-aided RIS Communications based on SWIPT" in 2022 IEEE Wireless Communications and Networking Conference (WCNC). [[Paper]](https://haoran-peng.github.io/Slides/peng1570767WCNC.pdf) [[Slides]]([https://github.com/Haoran-Peng/Haoran-Peng.github.io/blob/gh-pages/Slides](https://haoran-peng.github.io/Slides/EH_UAV_RIS.pdf) [[Video]](https://www.bilibili.com/video/BV1jL4y1F7oA#reply112394783936)
The implementation of DDPG is based on this [tutorial](https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b).
The architecture of actor and critic nets are implemented via using a MLP (2 layers of 64 neurals).

> There are some limitations to this work. If you have any questions or suggestions, please feel free to contact me. Your suggestions are greatly appreciated.

## Citing
Please consider **citing** our paper if this repository is helpful to you.
**Bibtex:**
```
@INPROCEEDINGS{peng1570767WCNC,
  author={Peng, Haoran and Wang, Li-Chun and Li, Geoffrey Ye and Tsai, Ang-Hsun},
  booktitle={Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)}, 
  title={Long-Lasting {UAV}-aided {RIS} Communications based on {SWIPT}},
  address={Austin, TX},
  year={2022},
  month = {Apr.}
}
```
## Requirements
- Python: 3.6.13
- Pytorch: 1.10.1
- gym: 0.15.3
- numpy: 1.19.2
- matplotlib
- pandas

## Usage
#### Descriptions of folders
- The folder "DDPG-SingleUT-Time" is the source code for the time-domain EH scheme using DDPG.
- The folder "DDPG-SingleUT-Time-and-Space" is the source code for the two-domain (Time and Space) EH scheme using DDPG.
- The folder "Exhaustive-SingleUT-Time" is the source code for the time-domain EH scheme using Exhaustive Algorithm.
- The folder "Exhaustive-SingleUT-Time-and-Space" is the source code for the two-domain (Time and Space) EH scheme using  Exhaustive Algorithm.

#### Descriptions of files
For the Exhaustive Algorithm, the communication environment is impletemented in 'ARIS_ENV.py'.
For the DDPG, the communication environment is impletemented in 'gym_foo/envs/foo_env.py'.
You can change the dataset in 'gym_foo/envs/foo_env.py'.

#### Training phase
1. In the main.py, the switch of "Train" must be 'True' such as
```
14 Train = True # True for tranining, and False for testing.
```
2. python main.py

#### Testing phase
1. In the main.py, the switch of "Train" must be 'False' such as
```
14 Train = False # True for tranining, and False for testing.
```
2. python main.py
3. The harvest energy of each step and overall steps are saved in 'Test_Rewards_Records.csv' and 'Total_Reward_Test.txt', respectively.

#### The EH efficiency
The EH efficiency for each step can be calculated by:
```
 reward of each step / 0.02275827153828275
```
