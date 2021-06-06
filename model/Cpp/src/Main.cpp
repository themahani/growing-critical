/*
 * Developer: Ali Mahani
 */

#include "include/NeuralNetwork.h"
#include <iostream>


int main(void)
{
    NeuralNetwork my_system = NeuralNetwork(1, 10, 0.05);
    my_system.print_dist_mat();

    return 0;
}
