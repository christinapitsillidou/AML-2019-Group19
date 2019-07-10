# Part 1: Gradient Descent
## Why is gradient descent important in machine learning?


The purpose of machine learning is to find a function that makes the best predictions. In order to determine that, the computer needs to compare numbers to see which function’s predictions are closer to the real values.


Let's consider finding the optimal prediction model as a shooting game. The shoot which has the shortest distance from the target will be the best shoot. In machine learning, we call the function of the distance a "loss function".


Minimizing this loss function can be complicated, especially for computationally complex functions or when the whole picture of the function is not known. Trying every combination of variables to find the smallest result of the loss function could be the best way but it is very slow. For this reason, ‘gradient descent’ algorithms were developed in order to make this process more efficient (find optimum prediction function by minimizing its loss functions).


## How does plain vanilla gradient descent work?


Gradient descent is a method to move from a point of the loss function to another with smaller value until we reach the minimum. In plain vanilla gradient descent, we move opposite to the direction of the slope and the step size is multiplied by the gradient(slope).


The word "descent" means going downhill. In 2-dimensions, if the slope is positive(upper-right direction) then we need to move to the smaller number(left direction) and vice versa (See figure1). 


The "gradient" means the slope. By considering the "U" shape of the graph as a loss function, we can see that to find the smallest point quicker, we need to move more when the slope is steeper(See figure2).


Finally, we need to consider the step size. If it is too small, then a lot of time is needed to find the minimum and if it is too big, then we could completely miss it(See figure3).


## How does plain vanilla gradient descent work?
