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

        int MAX_TIME;   // maximum time for isi
        std::vector<double> time_ax;    // time axis for calculating cumulative pdf
        std::vector<double> f_ax;       // list of firing rates to calculate cpdf
        std::vector< std::vector<double> > cpdf_matrix;     // initialized cpdf s.
        int next_neuron_spike;

        std::vector<int> spike_history;


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
                return std::pow((-d + r1 + r2) * (-d - r1 + r2) *
                        (-d + r1 - r2) * (d + r1 + r2), 0.5) * 0.5;
        }
    public:
        NeuralNetwork(int sys_size, int pop, double f_zero, double time_step,
                double g, double decay_time, double f_sat, double r_dot)
            : population(pop), size(sys_size), f0(f_zero), _h(time_step),
            g(g), tau(decay_time), f_sat(f_sat), K(r_dot), MAX_TIME(500), next_neuron_spike(0)
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

            /* Initialize time_ax from 0 to MAX_TIME with time step _h */
            for (int i = 0; i <= MAX_TIME; i += _h) {
                time_ax.push_back(i);
            }

            /* Initialize f_ax with 0 to 50 with steps 0.1 */
            for (int f = 0; f <= 50; f += 0.1) {
                f_ax.push_back(f);
            }

            /* Initialize cpdf_matrix */
            size_t length = f_ax.size();
            for (int i = 0; i < length; i++) {
                cpdf_matrix.push_back(calc_cpdf(calc_pdf(time_ax, f_ax[i])));
            }

        }

        /*
         * Calculate the probability distribution function for inter-spike intervals
         */
        std::vector<double> calc_pdf(std::vector<double> t, double f_rate)
        {
            std::vector<double> pdf;
            size_t size = t.size();
            pdf.reserve(size);
            for (int i = 0; i < size; i++) {
                pdf.push_back((f0 - (f_rate - f0) * std::exp(- t[i] / tau)) * std::exp(- f0 * t[i] - tau * (f_rate - f0) * (1 - std::exp(- t[i] / tau))));
            }
            return pdf;
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
         * Find index of nearest value in array
         */
        int nearest_value(std::vector<double> arr, double value)
        {
            int index = 0;
            while(value < arr[index])
                ++index;

            if (index == arr.size())
                return -1;
            else if (arr[index] - value > value - arr[index-1])
                return index - 1;
            else
                return index;
        }

        /*
         * Find next spike time and the neuron that spikes
         */
        double next_spike_time()
        {
            double earliest_spike = MAX_TIME;
            for (int i = 0; i < population; i++) {
                int f_index = nearest_value(f_ax, neuron_arr[i].firing_rate);

                double random = std::rand();    // generate random between 0 and 1
                LOG("random number between 0 and 1: " << random);
                double next_spike_index = nearest_value(cpdf_matrix[i], f_ax[f_index]);

                if (time_ax[next_spike_index] < earliest_spike)
                {
                    next_neuron_spike = i;
                    earliest_spike = time_ax[next_spike_index];
                }
            }
            spike_history.push_back(next_neuron_spike);
            return earliest_spike;
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
