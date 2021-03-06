import gym
import ppo
import sys
import numpy as np
import tensorflow as tf
import gym_wrapper


ENV = 'CartPole-v0'

if __name__ == "__main__":

    env = gym.make(ENV)

    with tf.Session() as sess:
        training = None
        if "-t" in sys.argv:
            training = True
        else:
            training = False


        actor = ppo.PPO(4,
                        2,
                        gamma=0.90,
                        lam=0.90,
                        clip_param=0.1,
                        horizon=5,
                        optim_epoch=4,
                        lr=0.0001,
                        lr_decay=0.001,
                        storage=10,
                        value_coefficient=1,
                        entropy_coefficient=0.001,
                        training=training,
                        continous=False)


        saver = tf.train.Saver()
        if "-n" in sys.argv:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "model/cartpole")
            print("Restored...")

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                gym_wrapper.play(env, actor)
            else:
                print("Training")
                gym_wrapper.train(env, actor, 1000000, render=False)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/cartpole")

