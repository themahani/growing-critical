#pragma once

/*
 * The highest level of abstraction for this project
 * is the neural network. This includes running the
 * simulation.
 */

#include "Neuron.h"
#include <bits/stdc++.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
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
            else if (d < r2 - r1 && r1 < r2)    // if circle 1 inside circle 2
                return M_PI * r1 * r1;
            else if (d < r1 - r2 && r1 >= r2)   // if circle 2 inside circle 1
                return M_PI * r2 * r2;
            else    // if have intersection
                {
                double part1 = std::pow(r1, 2) * std::acos((d*d + r1*r1 - r2*r2) / (2*d*r1));
                double part2 = std::pow(r2, 2) * std::acos((d*d - r1*r1 + r2*r2) / (2*d*r2));
                double part3 = std::sqrt((-d + r1 + r2) * (d - r1 + r2) * (d + r1 - r2) * (d + r1 + r2));
                return part1 + part2 - part3;
                }
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

            // initilize fired
            for (int i = 0; i < population; ++i) {
                fired.push_back(0);
            }
        }

        /*
         * Calculate the probability distribution function for inter-spike intervals
         */
        double calc_pdf(double t, double f_rate)
        {
            return (f0 - (f_rate - f0) * std::exp(- t / tau)) * std::exp(- f0 * t - tau * (f_rate - f0) * (1 - std::exp(- t / tau)));
        }

        std::vector<double> calc_cpdf(std::vector<double> pdf)
        {
            size_t length = pdf.size();
            std::vector<double> cpdf;
            cpdf.reserve(length);
            for (int i=0; i < length; ++i)
            {
                for (int j = 0; j < i; j++) // calculate cumulative sum
                {
                    cpdf[i] += pdf[j];
                }
            }

            return cpdf;
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
                for (int j = 0; j < i; ++j)
                {
                    mutual_area[i][j] = func(dist_mat[i][j],
                        neuron_arr[i].get_radius(), neuron_arr[j].get_radius());
                    mutual_area[j][i] = mutual_area[i][j];
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

        /*
         * output the position and radius to a file
         *
         * returns 0 is successful and 1 if failed
         */
        bool output_neuron_data(std::string run_name)
        {
            std::fstream output;    // create file stream

            output.open(std::string("data/") + run_name + std::string("_neuron_data.csv"), std::ios::out);   // open file to write data

            if (!output.is_open())  // report if failed and close
            {
                std::cerr << "[fstream]: Couldn't make file: " << strerror(errno) << std::endl;
                return errno;
            }
            else
            {
                output << "x, y, radius," << std::endl; // titles

                for (auto neuron : neuron_arr)  // for neuron in network
                {
                    output << neuron.get_x() << ", " << neuron.get_y() << ", "
                        << neuron.get_radius() << ", " << std::endl;    // data of each neuron
                }
            }
            output.close(); // close file and flush fstream
            return 0;
        }

        /*
         * calculate the mean intersection area
         */
        std::vector< double >  calc_mean_area()
        {
            std::vector< double > mean_area_vector;

            for (int j = 0; j < population; ++j)
            {
                double mean_area;
                for (int i = 0; i < population; ++i)
                {
                    mean_area += mutual_area[j][i];
                }

                mean_area_vector.push_back(mean_area / population);
            }

            return mean_area_vector;
        }

        void log_radius(std::fstream& output)
        {
            for (auto neuron : neuron_arr)
            {
                output << neuron.get_radius() << ", ";
            }
            output << std::endl;
        }

        /*
         * simulate the system for the given duration
         * and take samples of mean area intersection
         * of each neuron and store them in a file.
         */
        void simulate_system(double duration, std::string run_name)
        {
            std::fstream output;    // create file stream
            std::fstream radius_out;    // create file stream for radius

            output.open(std::string("data/") + run_name + std::string("_mean_area_intersection.csv"), std::ios::out);   // open file to write data
            radius_out.open("data/" + run_name + "_radius.csv", std::ios::out);     // open file to write

            if (!output.is_open() || !radius_out.is_open())  // report if failed and close
            {
                std::cerr << "[fstream]: Couldn't make file: " << strerror(errno) << std::endl;
            }
            else
            {
                int rep = int(duration / _h);   // find the number of time steps needed
                int interval = int(rep / 1000); // find interval for between each sampling
                if (interval == 0)        // if interval == 0
                {
                    for (int j = 0; j < rep; ++j) {    // run interval times
                        std::cout << "\r Progress: " << j / rep * 100 << "%" << std::flush;   // report progress

                        timestep(); // evolve for 1 time step
                        std::vector<double> means = calc_mean_area();   // sample mean_area_intersection

                        for (double mean : means)
                            output << mean << ", "; // write means to ouput file
                        output << std::endl;    // new line for new sampling

                        log_radius(radius_out);
                    }
                    std::cout << std::endl;     // new line after end of progress report
                }
                else
                {
                    for (int i = 0; i < 1000; ++i) {
                        std::cout << "\r Progress: " << i / 10.0 << "%" << std::flush;    // report probress

                        for (int j = 0; j < interval; ++j) {    // run interval times
                            timestep();
                        }

                        std::vector<double> means = calc_mean_area();   // sample mean_area_intersection
                        for (double mean : means)
                            output << mean << ", "; // write means to ouput file
                        output << std::endl;    // new line for new sampling

                        log_radius(radius_out);
                    }
                    std::cout << std::endl;     // new line after end of progress report
                }
            }
            output.close(); // close file and flush fstream
            radius_out.close(); // close file and flush fstream
        }

};
