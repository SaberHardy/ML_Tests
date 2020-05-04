"""Step 1 -->  Initialize parameters"""
cur_x = 3 #The algorithm starts at x=3
rate = 0.01  # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1
max_iterations = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x:2*(x+5) #Gradient of our function
# print(f"the value of df is: {df}")
"""Step 2 --> Run a loop to perform gradient descent """
while previous_step_size > precision and iters < max_iterations:
    prev_x = cur_x #store the curent value x  in previous
    cur_x = cur_x -rate * df(prev_x)
    previous_step_size = abs(cur_x - prev_x) #change in x
    iters = iters +1
    print("Iteration",iters,"\nX value is",cur_x) #Print iterations

print("The local minimum occurs at", cur_x)

