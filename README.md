# PPO 
  * Based on https://arxiv.org/pdf/1707.06347.pdf
  * RUN: python3 pendelum\_demo.py -p
  * TRAIN:  python3 pendelum\_demo.py -n -t


Implemented for Both Continous and Discrete problems, works great for continous.
Has some issues with catastrophic forgetting on discrete problems, atleast the Cartpole Environment.
The GIF below was from carefully tuning the parameters and stopping the training when i saw that it peaked, had i let it run
for a few more iterations it would probably crash. Any pointers how to make the discrete training more stable would be super cool! :) 


<a href="https://giphy.com/gifs/jxa5HFQeS3CLO2Sxdm"> <img src="https://media.giphy.com/media/jxa5HFQeS3CLO2Sxdm/giphy.gif" title="Pendelum demo"/></a>

![Imgur](https://i.imgur.com/vxiH7GY.png)



<a href="https://giphy.com/gifs/9AIdZ1IdJfih5t8slt"> <img src="https://media.giphy.com/media/9AIdZ1IdJfih5t8slt/giphy.gif" title="Cartpole demo"/></a>
