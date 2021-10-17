# The Model Implemented in C++
In this model, I created a class `Neuron` and a class `NeuralNetwork` to act
as the two levels of abstraction. The goal is to recreate the model presented
in the article **Growing Critical: Self-Organized Criticality in a Developing Neural System**.

## Execution
I have written a `Makefile` for this project.
so in order to run the simulation, simply run
`make`
and then run the generated binary file using
`./GrowingCritical`
for linux and
``./GrowingCritical.exe``
for windows.

## Directory Structure
The `src` directory contains the source code.
The `obj` directory contains the object files and temporary files
like the generate assembly code, etc.
The `data` directory contains the resulting data files.
The `res` directory contains the resulting plots from data analysis.
