import tensorflow as tf
from dqn import DQN

n = DQN((2,4,2))
x_train = [[0,0], [0,1], [1,0], [1,1]]
y_train = [[0,1], [1,0], [1,0], [0,1]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

n.set_session(sess)

for epoch in range(1,3000):
    cost, _ = n.update(x_train, y_train)
    if epoch % 10 == 0:
        print("COST", cost)

print(n.predict(x_train))

sess.close()
