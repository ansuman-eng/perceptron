import numpy as np
import tensorflow as tf

xarray=[[0,0],[0,1],[1,0],[1,1]]
yarray=[[0],[1],[1],[0]]

x_data=np.array(xarray)
y_data=np.array(yarray)

n_input=2;
n_hidden=10;
n_output=1;

learning_rate = 0.1
epochs = 10000

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden]))
W2 = tf.Variable(tf.truncated_normal([n_hidden, n_output]))

b1= tf.Variable(tf.truncated_normal([n_hidden]), name="Bias1")
b2= tf.Variable(tf.truncated_normal([n_output]), name="Bias2")

L2 = tf.nn.sigmoid(tf.matmul(X,W1)+b1)
hy = tf.nn.sigmoid(tf.matmul(L2,W2)+b2)

cost = tf.losses.mean_squared_error(Y,hy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for step in range(epochs):
        session.run(optimizer, feed_dict={X : x_data, Y : y_data})

        if(step%1000==0):
            print(session.run(cost, feed_dict={X: x_data, Y:y_data}))

    answer=tf.equal(tf.floor([hy+0.5]),Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))

    print(session.run([hy],feed_dict={X: x_data, Y:y_data}))
    print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data})*100, "%")
    
            
        

    
        
