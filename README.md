# Multilayered-Perceptron-Network-for-AI
Project for AI on MLP based Neural Networks.


The project was prepared as an assignment component in the University AI Course. The dataset consisted of employee data of 
an MNC which included attributes such as age, salary, gender, place of work, status (still working/resigned). The aim was to 
make a multi layer neural network based on Perceptrons to predict that given an employee information whether he/she is likely
to quit or to stay in. 

The script includes one-layer deep network (primarily because more layers only added to the computation time without much improvement
in the performance. About 1% improvment in accuracy, could be traded off for faster computation). Nodes were kept at a limited amount of 
5 in the hidden layer (also because of the same reason). The results were then subjected to 5-fold and 10-fold cross validation and gave
an approximate accuracy of 79.63 % in both the validations. 

Due to the validations in the script, it has an average run-time of 15 mins+. Fire up the script and wait for it to finish and 
show the final accuracy after each validation fold. 
