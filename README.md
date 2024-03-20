# Generalist Neural Algorithmic Learner([Report](final_report.pdf))

## Introduction
The Generalist Neural Algorithmic Learner is a project aimed at developing a graph neural network capable of learning and executing a wide range of algorithms. Inspired by existing research, our goal was to create a single model that can solve multiple algorithmic tasks, generalizing beyond the distribution it was trained on. This project represents a significant step forward in demonstrating how reasoning abilities can be applied to diverse control flows and tasks.

## Authors
- Ashwin Daswani, Boston University (ashwind@bu.edu)
- Duc Minh Nguyen, Boston University (nguymi01@bu.edu)
- Rohan Sawant, Boston University (rohan16@bu.edu)

## Project Overview
The Generalist Neural Algorithmic Learner is designed to learn the execution trace of algorithms given an intermediate state. Initially, the model is trained on single-task models for various algorithmic tasks. Subsequently, it is trained on multi-algorithmic tasks to evaluate its performance against the single-task models.

### Features
- **Graph Neural Networks**: Utilizes GNNs to learn and execute classical algorithmic tasks.
- **Single and Multi-task Learning**: Supports both single-task and multi-task experiment settings.
- **Model Improvements**: Includes various enhancements such as randomized position scalar, static hint elimination, encoder initialization, gradient clipping, and soft hint propagation.

## Installation
To run the project, ensure you have the following dependencies installed:

1. **JAX** (Note: JAX is not supported on Windows)
2. **NumPy**
3. **TensorFlow**
4. **CLRS from DeepMind**

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```



## Repository Structure
- `/models`: Contains the trained models and their parameters.
- `/experiments`: Includes Jupyter notebooks or scripts used for single-task and multi-task experiments.
- `/src`: Houses the source code for the Generalist Neural Algorithmic Learner.
- `/data`: Dataset used for training and evaluation, based on the CLRS30 benchmark.

## Usage
To use the Generalist Neural Algorithmic Learner, navigate to the `/src` directory and follow the instructions provided there for running the various experiments.

## Acknowledgments
This project was inspired by the work on the generalist neural algorithmic learner by DeepMind and other related research in the field of neural algorithmic reasoning.

## License
This project is open-source and available under the MIT license.




