## Evolutionary Agent Training in the Snake Game

### Project developers
- Tarubarova Nadezhda  
- Pokhodyaeva Polina  
- Gusev Saveliy  

### Project Idea
We are developing an artificial agent that learns to play the Snake game using evolutionary algorithms. The agent utilizes a genetic algorithm that evolves through mutations and the selection of the best strategies. Our goal is to improve the existing algorithm by enhancing training efficiency and introducing more advanced tasks for the agent.

### Method/Technique
- **Genetic Algorithm (GA)**: A population of agents evolves using evolutionary mechanics (selection, mutation, crossover).
- **Fitness Function**: Evaluates agents based on the snake’s length.
- **Enhancements**: We plan to modify mutation parameters and test different selection strategies.

### Dataset Description
There is no fixed dataset in our project since training occurs through an evolutionary algorithm. However, we collect game states during the training process.

**Example Data:**
- Snake and food coordinates  
- Movement direction  
- Snake length  
- Action (left, right, forward)  

### Key Changes
- **Activation Function**: Replaced with a sigmoid activation function in the `activate()` and `sigmoid()` methods.
- **Mutation**: Improved by introducing random weight changes with a small deviation.
- **Bias Addition**: Implemented in the `addBias()` method to enhance neural network flexibility.
- **Crossover**: Introduced a method for combining weights from two neural networks.
- **Output Method**: Developed `output()` to process input data through all neural network layers, applying the activation function.

### Repository
We are using an existing Snake AI repository on GitHub: **SnakeAI**

### References
- Snake AI (GA) GitHub Repository  
- Genetic Algorithm  
- Genetic Algorithm Applied to Games  
- Application of Genetic Algorithms in War Games  
- Playing Games with Genetic Algorithms  
- Application of Bio-Inspired Methods in Distributed Gaming Systems  
- Game Theory in a Bio-inspired Model of Computation  

