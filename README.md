# Computational Intelligence for Optimization
**University:** NOVA Information Management School<br>
**Program:** Master’s Degree Program in Data Science and Advanced Analytics<br>
**Academic Year:** 2022/2023<br>

**Students of Group 17:** 
- Iryna Savchuk (20211310)
- Cátia Parrinha (20201320)

## Project Details
### Task Description 
Employ Genetic Algorithms(GAs) to optimize the weights of a single-layer neural network (NN) for image classification purposes.

### Dataset Details
The MNIST dataset (a classic in the Machine Learning community) is a large dataset of handwritten digits assembled by the National Institute of Standards and Technology in the 1990s. This dataset contains 60,000 training images and 10,000 test images, all of the size 28x28 pixels and belonging to 10 classes.

Sample images from MNIST test dataset (picture taken from Wikipedia):
![mnist examples](https://github.com/iryna-savchuk/CI4O_project/blob/master/pictures/MnistExamples.png?raw=true)

### Project Objective
Students are asked to apply their knowledge about GAs to solve a specific optimization problem.

The steps expected to be done in the project:
- Implementation of several selection methods
- Implementation of several mutation operators
- Implementation of several crossover operators

The final goal is to illustrate and compare how various implementations differ in terms of performance. The decisions made should be based on statistical validation (meaning each selected configuration must be run about 100 times so that the performance of the different configurations could be compared statistically). 

### NN Architecture 
The primary focus of the project is on the GA optimization process, so it is sufficient to design a basic NN, with only one hidden layer. 
For example, the following basic architecture is suggested in "Deep Learning with Python", 2nd Edition (2021), by François Chollet:
<img src="https://github.com/iryna-savchuk/CI4O_project/blob/master/pictures/acrhitecture.png" alt="architecture" width="600">
