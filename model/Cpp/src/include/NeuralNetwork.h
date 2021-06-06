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


class NeuralNetwork
{
    private:
        const int population;   // how many neurons in the network
        int size;       // size of the lattice
        double f0;      // base firing rate of the system
        std::vector<Neuron> neuron_arr;     // an array of all the neurons
        std::vector< std::vector< double > > dist_mat;  // distance matrix
    public:
        NeuralNetwork(int sys_size, int pop, double f_zero)
            : population(pop), size(sys_size), f0(f_zero)
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
                            std::pow(neuron_arr[i].get_x() - neuron_arr[j].get_x(), 2) +
                            std::pow(neuron_arr[i].get_y() - neuron_arr[j].get_y(), 2));
                }
                dist_mat.push_back(row);
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
                    dist_mat[i][j] = std::pow(neuron_arr[i].get_x() - neuron_arr[j].get_x(), 2) +
                        std::pow(neuron_arr[i].get_y() - neuron_arr[j].get_y(), 2);
                }
            }
        }

        /*
         * print the distance matrix of the neurons to stdout
         */
        void print_dist_mat() const
        {
            for (auto row : dist_mat) {
                for (auto item : row) {
                    std::cout << item << ", ";
                }
                std::cout << std::endl;
            }
        }

        const std::vector< std::vector< double > >* const get_dist_mat() const
        {
            return &dist_mat;
        }
};
