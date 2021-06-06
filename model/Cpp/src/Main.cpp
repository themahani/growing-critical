/*
 * Developer: Ali Mahani
 */

#include "include/NeuralNetwork.h"
#include <iostream>


int main(void)
{
    NeuralNetwork my_system = NeuralNetwork(1, 10, 0.05);
    my_system.calc_mutual_area();
    my_system.print_matrix(*my_system.get_mutual_area());

    return 0;
}
