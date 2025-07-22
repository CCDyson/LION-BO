#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv

import numpy as np
import pandas as pd
from   skopt import gp_minimize
from skopt.space import Real, Integer
import random as rndm
import math as mth

import Particle as Prtcl
import Beam     as Bm
import BeamLine as BL
import BeamIO   as bmIO
import LION_EnvOptimisation_slimmed_down as EO
import BeamLineElement      as BLE
import UserFramework        as UsrFw
import pickle
import matplotlib.pyplot as plt

#This script is a rewording of the UserAnalysis.py script that you should have in your 03-Scripts folder
# It will be what we run to optimise the beamline
# It is also useful as an example of how to use/rewrite the UserAnalysis.py script to get 
# the desired output.


# The cost function that Josie wrote for optimising the position of PMQs
# This needs to be modified to take the tilts and shifts of PMQs as the parameters
# and instead of using the transmission to calculate the cost, it needs to be ncc

def cost_function(params, nEvts, response_funcs, exp_data):

    # define the parameters for optimization
    x1, y1, a1, b1, g1, x2, y2, a2, b2, g2  = params
 

    print('x shift 1:', x1)
    print('y shift 1:', y1)
    print('x shift 2:', x2)
    print('y shift 2:', y2)
    print('alpha tilt 1:', a1)
    print('beta tilt 1:', b1)
    print('gamma tilt 1:', g1)
    print('alpha tilt 2:', a2)
    print('beta tilt 2:', b2)
    print('gamma tilt 2:', g2)
    

    #.. ----> Instanciate user analysis:
    iEO = EO.UserAnal.getUserAnalInstances()[0]
    
    # Clears the beamline and particles
    BL.BeamLine.cleaninstance()
    BLE.BeamLineElement.cleaninstances()
    Prtcl.Particle.cleanAllParticles()


    ibmIOr = iEO.ibmIOr
    inputfile = ibmIOr.getdataFILE().name
    ibmIOr.getdataFILE().close()
    bmIO.BeamIO.cleanBeamIOfiles()
    
    # Collects source particles from the input file
    ibmIOr = bmIO.BeamIO(None, inputfile)
    EndOfFile = ibmIOr.readBeamDataRecord()

    # Builds the beamline with the specified parameters
    # Here we need to set the drifts to their fixed values 
    # and include the lines usually apparent in the csv of the beamline that describe
    # the tilts and shifts of the PMQs.
    startBLEparams = iEO.getstartBLEparams()
    iEO.setBLEparams([
        [startBLEparams[0][0], 0.04694],
        [startBLEparams[1][0], 1,0.003,0.0015],
        [startBLEparams[2][0], 0,0.005,x1,y1,a1,b1,g1],
        [startBLEparams[3][0],  0.04, 332., x1,y1,a1,b1,g1],
        [startBLEparams[4][0], 0,0.005,x1,y1,a1,b1,g1],
        [startBLEparams[5][0], 0.03233],
        [startBLEparams[6][0], 0,0.005,x2,y2,a2,b2,g2],
        [startBLEparams[7][0], 0.02,318.5, x2,y2,a2,b2,g2],
        [startBLEparams[8][0], 0,0.005,x2,y2,a2,b2,g2],
        [startBLEparams[9][0], 1.76543],
        [startBLEparams[10][0], 0,0.03],
    ])
    # Sets the new complete beamline
    iEO.setBeamLine()
    # print(BL.BeamLine.getinstances())
    
    # Tracks the beam through the beamline
    nEvtGen = BL.BeamLine.getinstances().trackBeam(nEvts, \
                                        None,
                                        None, None, False)
    
    # This is the function that needs to be modified to calculate the ncc
    # The function itself is in the EnvelopeBayesianOptimisation.py file
    cost = iEO.calc_cost(response_funcs, exp_data)

    
    return cost


# ------------------------------------------------------------------------------------------------------------


def main(argv):


# Checks files are given as arguments and if they are, that they are valid
    Success, Debug, \
        beamspecfile, inputfile, bdsimFILE, outputfile, nEvts = \
        UsrFw.startAnalysis(argv)
    if not Success:
        print(" <---- Failed at UsrFw.startAnalysis, exit")
        exit(1)

    # ibmIOr, ibmIOw are handling classes for the input and output files, read and write respectively
    Success, ibmIOr, ibmIOw = UsrFw.handleFILES(beamspecfile, \
                                                inputfile, \
                                                outputfile, \
                                                bdsimFILE)
    if not Success:
        print(" <---- Failed at UsrFw.handleFILES, exit")
        exit(1)

    #.. ----> Instanciate user analysis:
    iEO = EO.UserAnal(Debug)
    iEO.ibmIOr = ibmIOr
        
    #.. ----> Instanciate extrapolate beam class and extrapolate:
    iexBm = Bm.extrapolateBeam(ibmIOr, nEvts, None, None)
    iexBm.extrapolateBeam()
    Prtcl.Particle.cleanParticles()
    

#--------  Got "out of the box" size of beam, now try and optimise:

# This is the main part of the script

    ibmIOr.getdataFILE().close()
    
    #Prints out the beamline elements
    # Checks the beamline looks correct
    print(BL.BeamLine.getinstances())
    
    # Defines the limits of the search space for the parameters
    # Will need to be changed to the limits of the tilts and shifts of the PMQs 
    space = [ \
              Real(-0.1, 0.1, name='x1'), \
              Real(-0.1, 0.1, name='y1'),
              Real(-0.1, 0.1, name='a1'),
              Real(-0.1, 0.1, name='b1'),
              Real(-0.1, 0.1, name='g1'),
              Real(-0.1, 0.1, name='x2'),
              Real(-0.1, 0.1, name='y2'),
              Real(-0.1, 0.1, name='a2'),
              Real(-0.1, 0.1, name='b2'),
              Real(-0.1, 0.1, name='g2'),
             ]
    
    
    
    # Collect some constants before running the optimisation
    experiment_files = ['1RCF3.1.csv', '1RCF6.1.csv', '1RCF8.2.csv', '1RCF9.9.csv', '1RCF11.4.csv', '1RCF12.7.csv', '1RCF13.9.csv']
    exp_data_layers = [pd.read_csv('RCF/'+f).to_numpy() for f in experiment_files]
    exp_data = np.stack(exp_data_layers, axis=-1) # shape (H, W, 7) 
    
    #Temporary fix to experimental data that will force it to be 50 x 50 bins. 
    # In future the experimental data will be provided in the correct format
    # This is a workaround to ensure the optimisation works with the current data
    # This will need to be removed when the experimental data is provided in the correct format
    
    exp_data = exp_data[200:250, 50:100, :]  # Trim to 50x50x7 for now
    
    print('Shape of experimental data:', exp_data.shape)  # Should print (50, 50, 7)
    
    
    # This is to check the experimental data looks right. 
    # Uncomment when the right version of exp_data has been added
    
    # nrows, ncols, n_layers = exp_data.shape
    # extent = [0, ncols, 0, nrows]

    # # Plot each layer
    # for i in range(n_layers):
    #     values = exp_data[:, :, i]
    #     plt.figure(figsize=(6, 5))
    #     plt.imshow(values,
    #             extent=extent,
    #             origin='lower', aspect='auto', cmap='afmhot_r')
    #     plt.colorbar(label='Energy Deposited')
    #     plt.title(f'2D Histogram of Layer {i+1} ({experiment_files[i]})')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig(f"99-Scratch/2D_Histogram_Layer_{i+1}.png")
    


    response_functions = []  # Initialize as list
    for stack in range(7):
        # open a pickle file located in a 'stack-response-Functions' directory
        # the filename follows the pattern 'stack_{identifier}_response_function.pkl'
        with open(f"stack_response_functions/stack_{stack+1}_response_function.pkl", "rb") as file:
            response_func = pickle.load(file)
            response_functions.append(response_func)
            
    
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Initial guess for the parameters 
            
    test = cost_function(x, nEvts, response_functions, exp_data)
    
    return 0

        # This then runs the optimisation
    # The function to be minimised is the cost_function
    # The random state is set to None, so it will be random if you run it again
    # The acquisition function is set to EI (Expected Improvement)
    # The verbose is set to True, so it will print out the progress
    res = gp_minimize(lambda x: cost_function(x, nEvts, response_functions, exp_data), \
                      space, \
                      n_calls=2, \
                      n_initial_points=1, \
                      random_state=None, \
                      acq_func="EI", \
                      verbose=True)

    # Save results to CSV
    # I found this useful when there were only 2 parameters as I could plot the results
    # and see the parameter space myself
    results = []
    with open("99-Scratch/optimisation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "y1", "a1", "b1", "g1", "x2", "y2", "a2", "b2", "g2", "ncc"])  
    
        for x, cost_i in zip(res.x_iters, res.func_vals):
            x1, y1, a1, b1, g1, x2, y2, a2, b2, g2 = x
            writer.writerow([x1, y1, a1, b1, g1, x2, y2, a2, b2, g2, cost_i])
            results.append([x1, y1, a1, b1, g1, x2, y2, a2, b2, g2, cost_i])

    # This section is to run the optimisation multiple times with different random starting states
    # This is useful to get a better idea of the parameter space
    # and to avoid getting stuck in a local minimum
    # Might be useful to have the optimisation run above start on the default parameters of no shifts or tilts
    # and then run this section with random starting points to search the parameter space better
    # Or maybe use the first run to limit the search space and then run this section
    # These are things to consider but first we need to get the optimisation working
    # best_cost = res.fun
    # best_params = res.x
    random_states = rndm.sample(range(1000), 3)
    iteration_number = 0
    for idx, state in enumerate(random_states):
        iteration_number += 1
        print('************************************')
        print('Iteration state number __________ :', iteration_number)
        print('************************************')
        res = gp_minimize(lambda x: cost_function(x, nEvts, response_functions, exp_data), space, \
                      n_calls=10, \
                      n_initial_points=1, \
                      random_state=state, \
                      acq_func="EI", \
                      verbose=True)
        # Again the results are saved to a csv file. 
        # Again this is not needed for us but can be useful to see the results of the optimisation
        with open("99-Scratch/optimisation_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x1", "y1", "a1", "b1", "g1", "x2", "y2", "a2", "b2", "g2", "ncc"])  
        
            for x, cost_i in zip(res.x_iters, res.func_vals):
                x1, y1, a1, b1, g1, x2, y2, a2, b2, g2 = x
                writer.writerow([x1, y1, a1, b1, g1, x2, y2, a2, b2, g2, cost_i])
                results.append([x1, y1, a1, b1, g1, x2, y2, a2, b2, g2, cost_i])
        
        
    results_df = pd.DataFrame(results, columns=["x1", "y1", "a1", "b1", "g1", "x2", "y2", "a2", "b2", "g2", "ncc"])
    
    averaged_data = results_df.groupby(["x1", "y1", "a1", "b1", "g1", "x2", "y2", "a2", "b2", "g2"])['ncc'].mean().reset_index()
    
    best_params = averaged_data.loc[averaged_data['ncc'].idxmax()]


    # Extract optimal parameters
    optimal_x1 = best_params['x1']
    optimal_y1 = best_params['x2']
    optimal_a1 = best_params['a1']
    optimal_b1 = best_params['b1']
    optimal_g1 = best_params['g1']
    optimal_x2 = best_params['x2']
    optimal_y2 = best_params['y2']
    optimal_a2 = best_params['a2']
    optimal_b2 = best_params['b2']
    optimal_g2 = best_params['g2']
    optimal_cost = best_params['ncc']
    

    # Set optimal cost
    iEO.setCost(optimal_cost)
    print("Optimisation complete")
    print(f"Optimal parameters: \n x1={optimal_x1*1000}mm \n y1={optimal_y1*1000}mm \n a1={optimal_a1*1000}mm \n b1={optimal_b1*1000}mm \n g1={optimal_g1*1000}mm \n x2={optimal_x2*1000}mm \n y2={optimal_y2*1000}mm \n a2={optimal_a2*1000}mm \n b2={optimal_b2*1000}mm \n g2={optimal_g2*1000}mm")
    print("Optimal transmission:", optimal_cost)


#----  --------  --------  --------  --------  --------  --------  


    
"""
   Execute main"
"""
if __name__ == "__main__":
   main(sys.argv[1:])

sys.exit(1)
