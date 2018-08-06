import gym
import ppo
import sys
import numpy as np
import tensorflow as tf
import gym_wrapper


ENV = 'Pendulum-v0'


def action_modifier(action):
    return np.clip(action * 2, -2, 2)


if __name__ == "__main__":

    env = gym.make(ENV)

    with tf.Session() as sess:
        training = None
        if "-n" in sys.argv:
            training = True
        else:
            training = False

        actor = ppo.PPO(3,
                        1,
                        gamma=0.95,
                        clip_param=0.2,
                        batch=32,
                        optim_epoch=4,
                        lr_decay=0.001,
                        storage=32,
                        training=training,
                        continous=True)

        saver = tf.train.Saver()
        if "-n" in sys.argv:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "model/pendelum")
            print("Restored...")

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                gym_wrapper.play(env, actor, a_mod=action_modifier)
            else:
                gym_wrapper.train(env, actor, 1000000, a_mod=action_modifier, render=False)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/pendelum")

