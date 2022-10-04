#pragma once

/*
 * The implementation of a Neuron in the network as a
 * simple model. This class will be used as a new level
 * of abstraction.
 */

#define LOG(x) std::cout << x << std::endl

#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <iostream>


class Neuron
{
    private:
        double x_pos;
        double y_pos;
        double f0;
        double next_spike_time;

    public:
        double radius;
        double firing_rate;


        Neuron(int system_size, double f0)
            : firing_rate(f0), f0(f0), next_spike_time(0)    // list initializers
        {
            // initialize random seed
            // define maximum of rand to specify the accuracy of the double number
            int _RAND_MAX = 1000000;
            /* initialize random position and radius*/
            x_pos = (std::rand() % _RAND_MAX) / double(_RAND_MAX) * system_size;
            y_pos = (std::rand() % _RAND_MAX) / double(_RAND_MAX) * system_size;
            radius = (std::rand() % _RAND_MAX) / double(_RAND_MAX) * 0.05 * system_size;
        }

        /*! return x position */
        const double get_x() const
        {
            return x_pos;
        }

        /*! return y position */
        const double get_y() const
        {
            return y_pos;
        }

        /*! return radius */
        const double get_radius() const
        {
            return radius;
        }

        /* fire with a certain probability */
        bool fire(double& _h, double& K, double& f_sat)
        {
            int prob = firing_rate * _h;    // the equation for probability of firing a neuron
            fired =  (rand() % 1000) / 1000.0 < prob;   // fire with probability prob
            radius += K * _h;   // homogeneous radius update
            if (fired)
            {
                std::cout << "[Neuron]: fired... " << fired << std::endl;
                radius -= K / f_sat;    // in-homogeneous radius update
            }
            return fired;
        }
};
