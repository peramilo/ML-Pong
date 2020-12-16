# ML-Pong

An implementation of Deep Q-learning on simple 1 player pong game.


The goal is using machine learning to create and train an agent that will choose best action to perform based on 
the state of the game.
Agents decisions are based on expected discounted long-term reward for executing an action in a given state, value
which is called "Q-value". The object of Q-learning is estimating Q-values for optimal decision making policy that
maximizes future rewards.

In the game we have a paddle and a moving ball, the goal being to prevent the ball getting past the paddle and touch
the bottom wall. Only possible actions are moving the paddle either left or right.

Model used consists of 1 hidden layer, together with an input and output layer. 
Input consists of position of paddle's center and x and y coordinates of the ball.

TensorFlow and Keras modules were used to create and train deep learning model.
pygame module is used to display the attempts on the screen.
