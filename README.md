# AML_2019 Coursework, Part 1, Group 19
# Experiments with Gradient Descent


## Why is gradient descent important in machine learning?


The purpose of machine learning is to find a function that makes the best predictions. In order to determine that, the computer needs to find the function which makes the predictions with the smallest distance to the real values. In machine learning, we call the function of the distance a "loss function".


Minimizing this loss function can be complicated, especially for computationally complex functions or when the whole picture of the function is not known. Trying every combination of variables to find the smallest result of the loss function could be the best way but it is very slow.


## How does plain vanilla gradient descent work?

Gradient descent is a method to move from the point of the loss function to another with a smaller value until we reach the minimum. 


The word "descent" means going downhill. In 2-dimensions, if the slope goes upper-right direction then we need to move to the smaller left direction and vice versa. The "gradient" means the slope. By considering the "U" shape of the graph as a loss function, we can see that to find the smallest point quicker, we need to move more when the slope is steeper. 

![GD2](https://user-images.githubusercontent.com/52673999/60999676-a5283700-a353-11e9-9612-2a251de8f012.jpg)


In plain vanilla gradient descent, we move opposite to the direction of the slope and the step size is multiplied by the gradient(slope). Because it only uses the information of the current spot, it is easy to fall into a local minimum. So the initial point is important.

![vanilla](https://user-images.githubusercontent.com/52673999/61002684-f89d8380-a359-11e9-8deb-491e364c4b89.jpg)


Finally, we need to consider the step size. If it is too small, then a lot of time is needed to find the minimum and if it is too big, then we could completely miss it.

![stepsize png](https://user-images.githubusercontent.com/52673999/61003382-b7a66e80-a35b-11e9-9f6a-c006772f7aa8.jpg)


## Two modifications to plain vanilla gradient descent



| Parameter      | Explanation |
|----------------|-------------|
|`A`             | x           |
|`B`             | y           |
|`C`             | z           |
