#pragma once

/*
 * The implementation of a Neuron in the network as a
 * simple model. This class will be used as a new level
 * of abstraction.
 */

#define LOG(x) std::cout << x << std::endl

#include <stdlib.h>
#include <time.h>
#include <iostream>


class Neuron
{
    private:
        double _radius;
        double x_pos;
        double y_pos;

    public:
        Neuron(double radius, int system_size)
            : _radius(radius)
        {
            // initialize random seed
            std::srand(time(NULL));
            // define maximum of rand to specify the accuracy of the double number
            int _RAND_MAX = 100000;
            /* initialize random position */
            x_pos = (std::rand() % _RAND_MAX) / double(_RAND_MAX) * system_size;
            y_pos = (std::rand() % _RAND_MAX) / double(_RAND_MAX) * system_size;
        }
        /*! return x position */
        const double get_x_pos() const
        {
            return x_pos;
        }

        /*! return y position */
        const double get_y_pos() const
        {
            return y_pos;
        }

        /*! return radius */
        const double get_radius() const
        {
            return _radius;
        }

        void fire()
        {

        }
};
