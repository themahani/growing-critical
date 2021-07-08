/*
 * Developer: Ali Mahani
 */

#include "include/NeuralNetwork.h"
#include <iostream>
#include <chrono>


int main(void)
{
    auto start = std::chrono::high_resolution_clock::now();

    const char* run_name = "test_run3";

    /*
     * args:
     * network size: 1
     * population: 100
     * f0:  0.01
     * time step: 0.001
     * g: 500
     * tau = dacay_time: 0.01
     * f_sat: 2.0
     * K = r_dot: 0.0001
     */
    NeuralNetwork mySystem = NeuralNetwork(1, 100, 0.01, 0.001, 500, 0.01, 2.0, 0.0001);
    mySystem.output_neuron_data(run_name);
    mySystem.simulate_system(5000.0, run_name);

    auto stop = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds" << std::endl;

    return 0;
}
