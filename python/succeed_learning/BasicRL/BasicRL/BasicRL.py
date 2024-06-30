
import gym
import numpy as np

env = gym.make('FrozenLake-v1')
Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = .8
y = .95
n_episodes = 2000

rList = []

for i in range(n_episodes):
    s = env.reset()
    r_total = 0
    d = False
    j = 0

    while j < 99:
        j+=1
        a = np.argmax(Q[s,] + np.random.randn(1,env.action_space.n) * (1./(i+1)))
        s1,r,d = env.step(a)
        Q[s,a] = Q[s,a] + learning_rate*(r + y*np.max(Q[s1,:]) - Q[s,a])
        r_total += r
        s = s1
        if d == True:
            break

    rList.append(r_total)

print("Score / Time: " + str(sum(rList)/n_episodes))
print("Final Q-Table Values")
print(Q)