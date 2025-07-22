#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class UserAnal:
===============

  Dummy class created to help user develop their own analysis.

  Out of the box provides three "user hooks":

   UserInit: called at instanitation to allow user to initialise.

   UserAnal: called in the event loop to allow user to do whatever is needed
             for their analysis.

    UserEnd: called at the end of execution before termination to allow
             user to dump summaries, statistics, plots etc.

  Class attributes:
  -----------------
    instances : List of instances of Particle class
  __Debug     : Debug flag

      
  Instance attributes:
  --------------------
    
  Methods:
  --------
  Built-in methods __init__, __repr__ and __str__.
      __init__ : Creates instance of beam-line element class.
      __repr__: One liner with call.
      __str__ : Dump of constants

  Set methods:
     setDebug: set class debug flag
           Input: bool, True/False
          Return: None


Created on Tue 27Feb24: Version history:
----------------------------------------
 1.0: 27Feb24: First implementation

@author: kennethlong
"""

import io
import numpy as np

import Particle        as Prtcl
import BeamLine        as BL
import BeamLineElement as BLE
import PhysicalConstants as PC
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import os
print(f"Running Optimisation from: {os.path.abspath(__file__)}")

class UserAnal:
    instances  = []
    __Debug    = False


#--------  UserHooks:
    def UserInit(self):
        if self.getDebug():
            print("\n UserAnal.UserInit: initialsation")

        #--------  Initialise iteration records:
        self.setCost(None)
        self.setBLEparams([])

        #--------  Reading source distribution, so, now create
        #          PoPLaR beam line:
        
        # Sets default beam line parameters


        self.getBLEparams().append(["LION:1:Capture:Drift:1", 0.04694])
        self.getBLEparams().append(["LION:1:Capture:Elliptical:1",1,0.003,0.0015])
        self.getBLEparams().append(["LION:1:Capture:Circular:1",  0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.getBLEparams().append(["LION:1:Capture:Fquad:Shift:1",  0.04, 332., 0.0, 0.0, 0.0, 0.0, 0.0])
        self.getBLEparams().append(["LION:1:Capture:Circular:2",  0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.getBLEparams().append(["LION:1:Capture:Drift:2", 0.03233])
        self.getBLEparams().append(["LION:1:Capture:Circular:3",  0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.getBLEparams().append(["LION:1:Capture:Dquad:1", 0.02,318.5, 0.01, 0.0, 0.0, 0.0, 0.0])
        self.getBLEparams().append(["LION:1:Capture:Circular:4",  0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.getBLEparams().append(["LION:1:Delivery:Drift:1", 1.76543])
        self.getBLEparams().append(["LION:1:Delivery:Circular:1", 0, 0.03])


   
       


        self.setstartBLEparams(self.getBLEparams())
    
        self.setBeamLine()

    def UserEvent(self, iPrtcl):
        pass

    def UserEnd(self):
        pass

    def setBeamLine(self):
        if self.getDebug():
            print("\n Envelopeoptimisation.setBeamLine: start")
        #    print("     ----> BLEparams: \n", self.getBLEparams())
            
        #--------  Get beam line defined so far (by the input file) and reference particle:
        iBL      = BL.BeamLine.getinstances()
        refPrtcl = Prtcl.ReferenceParticle.getinstances()

        #.. Last beam line element:
        iLst = iBL.getElement()[-1]
        
        #--------  Reading source distribution, so, now create
        #          LION beam line:
    
        #.. Position of next beam-line element start:
        rStrt  = iLst.getrStrt() + iLst.getStrt2End()
        vStrt  = iLst.getvEnd()
        drStrt = np.array([0.,0.,0.])
        dvStrt = np.array([0.,0.,0.])


        #.. Add drift 1:
        iBLE    = BLE.Drift(self.getBLEparams()[0][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[0][1])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        #.. Add Elliptical collimator
        iBLE    = BLE.Aperture(self.getBLEparams()[1][0], rStrt, vStrt, drStrt, dvStrt,[self.getBLEparams()[1][1],self.getBLEparams()[1][2],self.getBLEparams()[1][3]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)
        
        
        #.. Focussing quadrupole:
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Aperture(self.getBLEparams()[2][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            [self.getBLEparams()[2][1],self.getBLEparams()[2][2],self.getBLEparams()[2][3],self.getBLEparams()[2][4],self.getBLEparams()[2][5],self.getBLEparams()[2][6],self.getBLEparams()[2][7]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)
        
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.FocusQuadrupole(self.getBLEparams()[3][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[3][1],self.getBLEparams()[3][2],
                            [self.getBLEparams()[3][3],self.getBLEparams()[3][4],self.getBLEparams()[3][5],self.getBLEparams()[3][6],self.getBLEparams()[3][7]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Aperture(self.getBLEparams()[4][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            [self.getBLEparams()[4][1],self.getBLEparams()[4][2],self.getBLEparams()[4][2],self.getBLEparams()[4][3],self.getBLEparams()[4][4],self.getBLEparams()[4][5],self.getBLEparams()[4][6],self.getBLEparams()[4][7]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)
        

        # Drift between quads
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Drift(self.getBLEparams()[5][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[5][1])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)
        
        #.. Defocussing quadrupole:
        
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Aperture(self.getBLEparams()[6][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            [self.getBLEparams()[6][1],self.getBLEparams()[6][2],self.getBLEparams()[6][3],self.getBLEparams()[6][4],self.getBLEparams()[6][5],self.getBLEparams()[6][6],self.getBLEparams()[6][7]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)
        

        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.DefocusQuadrupole(self.getBLEparams()[7][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[7][1],
                            self.getBLEparams()[7][2],[self.getBLEparams()[7][3],self.getBLEparams()[7][4],self.getBLEparams()[7][5],self.getBLEparams()[7][6],self.getBLEparams()[7][7]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Aperture(self.getBLEparams()[8][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            [self.getBLEparams()[8][1],self.getBLEparams()[8][2],self.getBLEparams()[8][3],self.getBLEparams()[8][4],self.getBLEparams()[8][5],self.getBLEparams()[8][6],self.getBLEparams()[8][7]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        #.. Add delivery drift 1:
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Drift(self.getBLEparams()[9][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[9][1])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        #.. Add delivery collimator
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Aperture(self.getBLEparams()[10][0],    \
                            rStrt, vStrt, drStrt, dvStrt, [self.getBLEparams()[10][1],self.getBLEparams()[10][2]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        
        #--------  Print at end:
        if self.getDebug():
            print(" UserAnal.UserInit: Beam line:")
            print(BL.BeamLine.getinstances())
            if BL.BeamLine.getinstances() == None:
                print("     ----> No beam line!  Quit.")
                exit(1)
    
    
#--------  Set/get methods
    def setAll2None(self):
        self.startBLEparams = []

    def setstartBLEparams(self, BLEparams):
        self.startBLEparams = BLEparams

    def setCost(self, Cost):
        self.Cost  = Cost
            
    def setBLEparams(self, BLEparams):
        self.BLEparams = BLEparams

    def getCost(self):
        return self.Cost
        
    def getstartBLEparams(self):
        return self.startBLEparams

    def getBLEparams(self):
        return self.BLEparams


#--------  "Built-in methods":
    def __init__(self, Debug=False):
        self.setDebug(Debug)
        if self.getDebug():
            print(' UserAnal.__init__: ', \
                  'creating the user analysis object object')

        UserAnal.instances.append(self)

        self.setAll2None()

        self.UserInit()

        if self.getDebug():
            print("     ----> New UserAnal instance: \n", \
                  UserAnal.__str__(self))
            print(" <---- UserAnal instance created.")
            
    def __repr__(self):
        return "UserAnal()"

    def __str__(self):
        self.print()
        return " UserAnal __str__ done."

    def print(self):
        print("\n UserAnal:")
        print(" ---------")
        print("     ----> Debug flag:", self.getDebug())
        return " <---- UserAnal parameter dump complete."

    
#--------  "Set method" only Debug
#.. Method believed to be self documenting(!)

    @classmethod
    def setDebug(cls, Debug=False):
        cls.__Debug = Debug
        if cls.__Debug:
            print(" UserAnal.setDebug: ", Debug)

    
#--------  "Get methods" only; version, reference, and constants
#.. Methods believed to be self documenting(!)

    @classmethod
    def getDebug(cls):
        return cls.__Debug

    @classmethod
    def getUserAnalInstances(cls):
        return cls.instances


#--------  Processing methods:
    def EventLoop(self, ibmIOw):

        nPrtcl = 0
        for iPrtcl in Prtcl.Particle.getParticleInstances():
            nPrtcl += 1
            if isinstance(iPrtcl, Prtcl.ReferenceParticle):
                iRefPrtcl = iPrtcl
                continue
            iLoc = -1

            self.UserEvent(iPrtcl)

            if isinstance(ibmIOw, io.BufferedWriter):
                iPrtcl.writeParticle(ibmIOw)
        
        Prtcl.Particle.cleanParticles()
        
    def histo_data(cls, x, y, ke, response_funcs):
        """
        Generate a 3D stack of 2D histograms representing energy deposition 
        in successive detector layers (e.g., RCF films), based on particle positions 
        and kinetic energies, filtered by energy thresholds.

        Parameters:
        - x, y: arrays of particle positions [m]
        - ke: array of particle kinetic energies [MeV]
        - response_funcs: list of vectorized functions, one per RCF film/layer

        Returns:
        - counts: 3D array (50, 50, n_layers) of normalized 2D histograms
        """

        # Energy thresholds (MeV) — min energy to register in each film
        stack_fn_cut = [3., 5.821401202856385, 7.997094322613386, 9.658781341633036, 
                        11.148286563497997, 12.435592394828737, 13.723555017136873]

        print('Getting histogram data')

        # Convert positions to millimetres
        x_mm = np.array(x) * 1000.0  # Convert from m to mm
        y_mm = np.array(y) * 1000.0  # Convert from m
        ke = np.array(ke)

        histograms = []
        xedgeses = []
        yedgeses = []

        for i, cut in enumerate(stack_fn_cut):
            # Filter particles above threshold
            mask = ke > cut
            if not np.any(mask):
                histograms.append(np.zeros((50, 50)))
                xedgeses.append(np.linspace(0, 1, 51))  # dummy edges
                yedgeses.append(np.linspace(0, 1, 51))  # dummy edges
                continue

            ke_filtered = ke[mask]
            x_filtered = x_mm[mask]
            y_filtered = y_mm[mask]

            # Compute deposited energy using the response function
            deposited = response_funcs[i](ke_filtered)

            # 2D histogram of deposited energy
            hist, xedges, yedges = np.histogram2d(
                x_filtered, y_filtered,
                bins=(50, 50),
                weights=deposited
            )

            # Normalize to total energy deposited in that film
            hist_norm = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            histograms.append(hist_norm)
            xedgeses.append(xedges)
            yedgeses.append(yedges)

        # Stack histograms into a 3D array (50, 50, n_layers)
        counts = np.stack(histograms, axis=-1)
        
        return counts, xedgeses, yedgeses
    
    def NCC(cls, sim_counts, exp_counts):
        """
        Compute the Normalized Cross-Correlation (NCC) for each layer (2D histogram)
        in two 3D stacks: experimental and simulated.

        Parameters:
        - exp_counts: ndarray of shape (H, W, N), experimental 2D histograms (N layers)
        - sim_counts: ndarray of shape (H, W, N), simulated 2D histograms

        Returns:
        - nccs: ndarray of shape (N,), NCC for each layer
        """
        
        if exp_counts.shape != sim_counts.shape:
            raise ValueError("Input shapes must match \n(exp_counts: {}, sim_counts: {})".format(exp_counts.shape, sim_counts.shape))

        height, width, num_layers = exp_counts.shape
        nccs = np.zeros(num_layers)

        for i in range(num_layers):
            exp_layer = exp_counts[:, :, i].astype(np.float64)
            sim_layer = sim_counts[:, :, i].astype(np.float64)

            exp_mean = np.mean(exp_layer)
            sim_mean = np.mean(sim_layer)

            exp_centered = exp_layer - exp_mean
            sim_centered = sim_layer - sim_mean

            numerator = np.sum(exp_centered * sim_centered)
            denominator = np.sqrt(np.sum(exp_centered**2)) * np.sqrt(np.sum(sim_centered**2))

            if denominator < 1e-12:  # Avoid division by zero. 
                # Most common case for us will be when no particles reach a layer and the histogram is all zeros.
                nccs[i] = -1.0  # Assign -1 for undefined NCC
            else:
                nccs[i] = numerator / denominator
                
        ncc = np.mean(nccs)  # Average NCC across all layers

        return ncc
    
    def plot_histograms(cls, histogram_counts, xedges, yedges):

        for i in range(histogram_counts.shape[-1]):
            print(f"Plotting histogram for layer {i+1}")
            values = histogram_counts[:, :, i]
            xedges_i = xedges[i]
            yedges_i = yedges[i]
            extent = [xedges_i[0], xedges_i[-1], yedges_i[0], yedges_i[-1]]

            plt.figure(figsize=(6, 5))
            plt.imshow(values.T,origin='lower', extent=extent, aspect='auto', cmap='afmhot_r')
            plt.colorbar(label='Energy Deposited')
            plt.title(f'Simulated 2D Histogram for Layer {i+1}')
            plt.xlabel('X (cm)')
            plt.ylabel('Y (cm)')
            plt.grid(True)
            plt.savefig(f'99-Scratch/sim_data_histogram_layer_{i+1}.png', dpi=300)

        return 0


    def calc_cost(cls, response_funcs, exp_data):
        """
        Calculate the cost (1 - normalized cross correlation) between simulated and experimental data.

        This method is typically called in the main script to compute how well the simulated particle
        transmission matches experimental measurements, expressed via a cost function suitable for minimization.

        Steps:
        - Retrieves all existing particle instances and their trace space through the beamline.
        - Extracts particle end positions (x, y) and kinetic energies (ke) after the final beamline element.
        - Uses provided response functions to convert kinetic energies to deposited energies in a detector stack.
        - Builds histograms of deposited energies spatially.
        - Compares simulated histograms with experimental histograms using normalized cross correlation (NCC).
        - Computes cost as 1 - NCC, where lower cost indicates better agreement.

        Parameters:
        - response_funcs : list of callables
            List of response functions for each detector layer. Each function maps kinetic energies to deposited energies.
        - exp_data : np.ndarray
            Experimental histogram data to compare against (shape should match histograms produced by `histo_data`).

        Returns:
        - cost : float
            A scalar cost value representing the mismatch between simulation and experiment.
            Returns 0 if no particles are available.

        Notes:
        - Assumes the particle mass for protons is retrieved internally.
        - Assumes beamline consists of 9 elements, and trace space data length and shape are checked accordingly.
        - Cleans particle instances from memory after processing.
        """
        # This is the function called in the main script to calculate the transmission
        
        pmass = PC.PhysicalConstants().getparticleMASS('Proton')  # MeV/c^2
        nEvts = len(Prtcl.Particle.getinstances())

        if nEvts == 0:
            # print("No particles available.")
            return 0

        Particles = Prtcl.Particle.getinstances()
        x=[]
        y=[]
        ke=[]
        num_beamline_elements = 11
        
        for iPrtcl in Particles:
            
            trace_space = iPrtcl.getTraceSpace()
            # Check on trace space to avoid IndexError
            # ie checks if the particle has reached the end of the beamline
            # and has a valid trace space
        
            if len(trace_space) > num_beamline_elements and len(trace_space[num_beamline_elements]) >= 3:
                phase = Prtcl.Particle.RPLCTraceSpace2PhaseSpace(trace_space[-1])

                phase = np.array(phase)
                x_end, y_end = phase[0, 0], phase[0, 1]
                px, py, pz = phase[1, 0], phase[1, 1], phase[1, 2]
                if pz is not None:
                    ke_end = np.sqrt(px**2 + py**2 + pz**2 + pmass**2) - pmass # rest energy
                    x.append(x_end)
                    y.append(y_end)
                    ke.append(ke_end)
            # Clean memory after saving particle
            Prtcl.Particle.cleanParticles()
            
        print("Particles Collected:", len(x))
        
        histogram_counts, xedges, yedges = cls.histo_data(x, y, ke, response_funcs)
        print("Histograms Created")
        
        cost = 1 - cls.NCC(histogram_counts, exp_data)
        
        print('*************************************************************')
        print(f"_________Total cost value_________: {cost:.2f}")
        print('*************************************************************')
        
        
        # Added section to plot the histograms to see if the parameters are changing the beam as expected
        cls.plot_histograms(histogram_counts, xedges, yedges)
        
        return cost
    




#--------  Exceptions:
class noReferenceParticle(Exception):
    pass

