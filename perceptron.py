import tensorflow as tf
import numpy as np
import csv

array=[]
xarray=[]
yarray=[]
print("Processing the input file.....")
with open("input1.csv") as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        array.append(row)

for i in range(1,10001):
    if(i%1000==0):
        print(i,"/ 10000")
    xarray.append(array[i][3:13])
    yarray.append(array[i][13:])
print(len(array))
print("We will start with 5 fold cross validation")
five_fold=0
for val in range(5):
    print("Fold ", val+1)
    start_index=(val)*len(array)//5
    end_index=(val+1)*len(array)//5

    x_test=np.array(xarray[start_index:end_index])
    y_test=np.array(yarray[start_index:end_index])

    x_data=np.array(xarray[0:start_index]+xarray[end_index:])
    y_data=np.array(yarray[0:start_index]+yarray[end_index:])

    #hyperparameters
    #Pilani =0, Goa =1, Hyderabad =2
    #Female = 1, Male =0

    n_input = 10
    n_hidden1 = 5
    n_output= 1
    learning_rate=0.3
    epochs = 1000

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    print("Placeholders for tensors created...")

    #weights
    W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden1]),name='W1')
    W2 = tf.Variable(tf.truncated_normal([n_hidden1, n_output]),name='W2')



    print("Randomized weights assigned...")

    #bias

    b1 = tf.Variable(tf.truncated_normal([n_hidden1]), name="Bias1")
    b2 = tf.Variable(tf.truncated_normal([n_output]), name="Bias2")

    print("Randomized biases assigned...")

    hidden1_out = tf.sigmoid(tf.add(tf.matmul(X,W1),b1))
    f_out = tf.sigmoid(tf.add(tf.matmul(hidden1_out,W2), b2))

    ################ Cost Function ##############
    cost = tf.losses.mean_squared_error(Y,f_out)

    ################ Optimizer #################
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    ############### Initialize, Accuracy and Run ###########
    # Initialize variables
    init = tf.global_variables_initializer()

    #Accuracy for the test set
    #accuracy = tf.reduce_mean(tf.square(tf.subtract(Y,f_out)))

    #Run
    with tf.Session() as session:
        session.run(init)

        for step in range(epochs):
            session.run(optimizer, feed_dict = {X: x_data, Y: y_data})
            c=session.run(cost, feed_dict = {X: x_data, Y: y_data})
            #print("Step number", step)
            if(step%100==0):
                print("Epoch number :", step+1, ", Cost :", c)

        answer=tf.equal(tf.floor(f_out+0.5),Y)
        accuracy = tf.reduce_mean(tf.cast(answer,"float"))
                              
        #print(session.run(cost, feed_dict={X: x_test, Y: y_test}))
        print(accuracy.eval({X: x_test, Y: y_test}) * 100)
        five_fold=five_fold+(accuracy.eval({X: x_test, Y: y_test}) * 100)


print("Average accuracy over all 5 folds was : ", five_fold/5)

print("We will start with 10 fold cross validation")
ten_fold=0
for val in range(10):
    print("Fold ", val+1)
    start_index=(val)*len(array)//10
    end_index=(val+1)*len(array)//10

    x_test=np.array(xarray[start_index:end_index])
    y_test=np.array(yarray[start_index:end_index])

    x_data=np.array(xarray[0:start_index]+xarray[end_index:])
    y_data=np.array(yarray[0:start_index]+yarray[end_index:])

    #hyperparameters
    #Pilani =0, Goa =1, Hyderabad =2
    #Female = 1, Male =0

    n_input = 10
    n_hidden1 = 5
    n_output= 1
    learning_rate=0.3
    epochs = 1000

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    print("Placeholders for tensors created...")

    #weights
    W1 = tf.Variable(tf.truncated_normal([n_input, n_hidden1]),name='W1')
    W2 = tf.Variable(tf.truncated_normal([n_hidden1, n_output]),name='W2')



    print("Randomized weights assigned...")

    #bias

    b1 = tf.Variable(tf.truncated_normal([n_hidden1]), name="Bias1")
    b2 = tf.Variable(tf.truncated_normal([n_output]), name="Bias2")

    print("Randomized biases assigned...")

    hidden1_out = tf.sigmoid(tf.add(tf.matmul(X,W1),b1))
    f_out = tf.sigmoid(tf.add(tf.matmul(hidden1_out,W2), b2))

    ################ Cost Function ##############
    cost = tf.losses.mean_squared_error(Y,f_out)

    ################ Optimizer #################
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    ############### Initialize, Accuracy and Run ###########
    # Initialize variables
    init = tf.global_variables_initializer()

    #Accuracy for the test set
    #accuracy = tf.reduce_mean(tf.square(tf.subtract(Y,f_out)))

    #Run
    with tf.Session() as session:
        session.run(init)

        for step in range(epochs):
            session.run(optimizer, feed_dict = {X: x_data, Y: y_data})
            c=session.run(cost, feed_dict = {X: x_data, Y: y_data})
            #print("Step number", step)
            if(step%100==0):
                print("Epoch number :", step+1, ", Cost :", c)

        answer=tf.equal(tf.floor(f_out+0.5),Y)
        accuracy = tf.reduce_mean(tf.cast(answer,"float"))
                              
        #print(session.run(cost, feed_dict={X: x_test, Y: y_test}))
        print(accuracy.eval({X: x_test, Y: y_test}) * 100)
        ten_fold=ten_fold+(accuracy.eval({X: x_test, Y: y_test}) * 100)


print("Average accuracy over all 10 folds was : ", ten_fold/10)
    





    
