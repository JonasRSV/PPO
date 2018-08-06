# PPO 
  * Based on https://arxiv.org/pdf/1707.06347.pdf

### Requirements
  * python 3.x

> INSTALL: pip3 install -r requirements.txt

> TRAIN: python3 pendelum\_demo.py -n -t

> PLAY: python3 pendelum\_demo.py -p


Summaries
---
Start Tensorboard on the summaries directory, create one if there is none then run:
> tensorboard --logdir=summaries



Works Great for continous problems! :) 


<a href="https://giphy.com/gifs/jxa5HFQeS3CLO2Sxdm"> <img width=350px src="https://media.giphy.com/media/jxa5HFQeS3CLO2Sxdm/giphy.gif" title="Pendelum demo"/></a>

![Imgur](https://i.imgur.com/vxiH7GY.png)


> Works less Great for discrete problems

The GIF below was from carefully tuning the parameters and stopping the training when i saw that it peaked, had i let it run
for a few more iterations it would probably crash. Any pointers how to make the discrete training more stable would be super cool! :) 


<a href="https://giphy.com/gifs/9AIdZ1IdJfih5t8slt"> <img width=350px src="https://media.giphy.com/media/9AIdZ1IdJfih5t8slt/giphy.gif" title="Cartpole demo"/></a>
