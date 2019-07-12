# AML_2019 Coursework, Part 1, Group 19
# Experiments with Gradient Descent


## Why is gradient descent important in machine learning?


The purpose of machine learning is to find a function that makes the best predictions. In order to determine that, the computer needs to find the function which makes the predictions with the smallest distance to the real values. This function of the distance is called a "loss function".


Minimizing this loss function can be complicated, especially for computationally complex functions or when the whole picture of the function is not known. Trying every combination of variables could be the best way but it is very slow.


## How does plain vanilla gradient descent work?

Gradient descent is a method to move from a point of the loss function to another with a smaller value until we reach the minimum. 


The word "descent" means going downhill. In 2-dimensions, if the slope goes to the upper-right direction, then we need to move to the left to get a smaller value and vice versa. The "gradient" means the slope. By considering the "U" shaped graph below as a loss function, we can see that to find the minimum quicker, a bigger move is needed when the slope is steeper. 

![GD5](https://user-images.githubusercontent.com/52673999/61009965-710d4000-a36c-11e9-81a8-8ff8e4444aaa.jpg)


In Plain Vanilla gradient descent, we move opposite to the direction of the slope and the step size is multiplied by the gradient(slope). Because it only uses the information of the current spot, it is easy to fall into a local minimum. So the initial point is important.

![vanilla2](https://user-images.githubusercontent.com/52673999/61004906-ee31b880-a35e-11e9-90f3-6b05db7d64e1.jpg)


Finally, we need to consider the step size. If it is too small, then a lot of time is needed to find the minimum and if it is too big, Plain Vanilla could completely miss it.

![stepsize8](https://user-images.githubusercontent.com/52673999/61007513-e83ed600-a364-11e9-9036-bed53afe0a01.jpg)


## Two variants of gradient descent

One of the problems of gradient descent is that it takes a long time to find a solution as it calculates all the possibilities for the slope in order to find the optimal next step. For Plain Vanilla, the direction and step size of the next step are problems that could be improved.


Momentum is the way of using previous information of the direction to find the next step. Therefore, the result looks more rounded than others like rolling a heavy ball. ADAM considers one more factor, step size, based on momentum theory. The step of ADAM looks like rolling a light ball.

![result6](https://user-images.githubusercontent.com/52673999/61088000-18eb4200-a42f-11e9-8aee-936078a73051.jpg)


From the starting point (1.5, 1.5), Momentum method does not have enough power to go through the small hill to find the global minimum. However, ADAM can find the global minimum because it also adjusts its step size. It takes more steps than other models but finds the right global minimum value.

![result2](https://user-images.githubusercontent.com/52673999/61085398-10433d80-a428-11e9-810d-4123ab4b0fed.jpg)


From the starting point (2.25,1.5), Momentum method has enough power to go through the small hill to find the global minimum. ADAM cannot find the global minimum because the initial point has steeper slope.

![result3](https://user-images.githubusercontent.com/52673999/61085430-294bee80-a428-11e9-9954-0ff8bc29ae32.jpg)
