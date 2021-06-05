/*
 * Developer: Ali Mahani
 */

#include "include/Neuron.h"
#include <iostream>


int main(void)
{
    Neuron my_neuron = Neuron(0.5, 1);
    std::cout << my_neuron.get_x_pos() << ", " << my_neuron.get_y_pos() << std::endl;
    std::cout << my_neuron.get_radius() << std::endl;

    return 0;
}
