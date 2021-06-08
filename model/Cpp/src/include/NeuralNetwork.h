#pragma once

/*
 * The highest level of abstraction for this project
 * is the neural network. This includes running the
 * simulation.
 */

#include "Neuron.h"
#include <bits/stdc++.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>


class NeuralNetwork
{
    private:
        const int population;   // how many neurons in the network
        int size;       // size of the lattice
        double f0;      // base firing rate of the system
        std::vector<Neuron> neuron_arr;     // an array of all the neurons
        std::vector< std::vector< double > > dist_mat;  // distance matrix
        std::vector< std::vector< double > > mutual_area;   // mutual area matrix
        double _h;      // duration of each time step
        int g;          // mutual area coefficient
        double tau;     // spike decay characteristic time
        double K;       // equation for R dot
        double f_sat;   // how much the f_i goes down
        std::vector< bool > fired;  // which neurons have fired

        /*
         * function to calculate the intersection area of
         * two circles
         */
        double func(double d, double r1, double r2)
        {
            if (d >= r1 + r2)    // if circles have no intersection
                return 0.0;
            else if (d < r1 - r2 && r1 < r2)    // if circle 1 inside circle 2
                return M_PI * r1 * r1;
            else if (d < r1 - r2 && r1 >= r2)   // if circle 2 inside circle 1
                return M_PI * r2 * r2;
            else    // if have intersection
                return std::pow((-d + r1 + r2) * (-d - r1 + r2) *
                        (-d + r1 - r2) * (d + r1 + r2), 0.5) * 0.5;
        }
    public:
        NeuralNetwork(int sys_size, int pop, double f_zero, double time_step,
                double g, double decay_time, double f_sat, double r_dot)
            : population(pop), size(sys_size), f0(f_zero), _h(time_step),
            g(g), tau(decay_time), f_sat(f_sat), K(r_dot)
        {
            /* initialize random seed */
            std::srand(time(NULL));
            /* initialize an array of neurons */
            for (int i = 0; i < population; ++i)
            {
                neuron_arr.push_back(Neuron(size, f0));
            }

            /* initialize the distance matrix */
            for (int i = 0; i < population; ++i) {
                std::vector<double> row;
                for (int j = 0; j < population; ++j) {
                    row.push_back(
                            pow(std::pow(neuron_arr[i].get_x() - neuron_arr[j].get_x(), 2) +
                                std::pow(neuron_arr[i].get_y() - neuron_arr[j].get_y(), 2), 0.5));
                }
                dist_mat.push_back(row);
            }

            /* initialize the mutual area matrix */
            for (int i = 0; i < population; ++i) {
                std::vector<double> row;
                for (int j = 0; j < population; ++j) {
                    row.push_back(0.0);
                }
                mutual_area.push_back(row);
            }

            // inicilize fired
            for (int i = 0; i < population; ++i) {
                fired.push_back(0);
            }
        }

        /*
         * print the position x,y of each neuron on a line
         */
        void print_pos() const
        {
            for (int i = 0; i < population; ++i)
            {
                std::cout << neuron_arr[i].get_x() << ", " <<
                    neuron_arr[i].get_y() << std::endl;
            }
        }

        /*
         * calculate the distance matrix of the neurons
         */
        void calc_dist_mat()
        {
            for (int i = 0; i < population; ++i) {
                for (int j = 0; j < population; ++j) {
                    dist_mat[i][j] = std::pow(std::pow(neuron_arr[i].get_x() - neuron_arr[j].get_x(), 2) +
                        std::pow(neuron_arr[i].get_y() - neuron_arr[j].get_y(), 2), 0.5);
                }
            }
        }

        /*
         * print the distance matrix of the neurons to stdout
         */
        void print_matrix(std::vector< std::vector< double > > matrix) const
        {
            for (auto row : matrix) {
                for (auto item : row) {
                    std::cout << item << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        const std::vector< std::vector< double > >* const get_dist_mat() const
        {
            return &dist_mat;
        }

        /*
         * calculate the mutual area of the neurons
         */
         void calc_mutual_area()
         {
             for (int i = 0; i < population; ++i)
             {
                for (int j = 0; j < population; ++j)
                {
                    mutual_area[i][j] = func(dist_mat[i][j],
                        neuron_arr[i].get_radius(), neuron_arr[j].get_radius());
                }
             }
         }

        const std::vector< std::vector< double > >* const get_mutual_area() const
        {
            return &mutual_area;
        }

        /*
         * evolve the system for 1 timestep
         * and update the radius and firing rate
         * of each neuron in the network
         */
        void timestep()
        {
            // fire neurons
            for (int i = 0; i < population; ++i)
            {
                fired[i] = neuron_arr[i].fire(_h, K, f_sat);
                neuron_arr[i].firing_rate += (f0 - neuron_arr[i].firing_rate) / tau * _h;   // homogeneous f_i update
                // in-homogeneous f_i update
                for (int j = 0; j < population && j != i; ++j)
                {
                    if (fired[j] == 1 && mutual_area[i][j] != 0)
                        neuron_arr[i].firing_rate += tau * g * mutual_area[i][j];
                }
            }

            calc_mutual_area();      // update mutual area

        }

        /*
         * Function that evolves the system for a
         * specified duration
         */
        void evolve(double duration)
        {
            int rep = int(duration / _h);   // find the number of time steps needed

            print_matrix(mutual_area);
            for (int i = 0; i < rep; ++i) {
                timestep();
            }
            print_matrix(mutual_area);
        }
};
