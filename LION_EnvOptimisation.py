#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class UserAnal:
===============
print(herehehrjbkj ee)
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
        self.getBLEparams().append(["LION:1:Delivery:Drift:2", 0.01])
        self.getBLEparams().append(["LION:1:Delivery:Circular:1", 0, 0.005])
        self.getBLEparams().append(["LION:1:Delivery:Drift:3", 0.02])

   
       


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
        print('here')
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

        #.. Add delivery drift 2:
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Drift(self.getBLEparams()[10][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[10][1])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)

        #.. Add delivery collimator
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Aperture(self.getBLEparams()[11][0],    \
                            rStrt, vStrt, drStrt, dvStrt, [self.getBLEparams()[11][1],self.getBLEparams()[11][2]])
        BL.BeamLine.addBeamLineElement(iBLE)
        refPrtclSet = refPrtcl.setReferenceParticle(iBLE)


        #.. Add delivery drift 3;
        rStrt  = iBLE.getrStrt() + iBLE.getStrt2End()
        iBLE    = BLE.Drift(self.getBLEparams()[12][0],    \
                            rStrt, vStrt, drStrt, dvStrt, \
                            self.getBLEparams()[12
                                                ][1])
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


    def calculate_transmission(cls):
        # This is the function called in the main script to calculate the transmission
        # I left this in for reference. We will change it to be calculate chi squared
        
        # Set the threshold for the beam size at the end of the beamline
        
        pmass = PC.PhysicalConstants().getparticleMASS('Proton')  # MeV/c^2
        nEvts = len(Prtcl.Particle.getinstances())

        if nEvts == 0:
            # print("No particles available.")
            return 0

        Particles = Prtcl.Particle.getinstances()
        for iPrtcl in Particles:
            
            trace_space = iPrtcl.getTraceSpace()
            # Check on trace space to avoid IndexError
            # ie checks if the particle has reached the end of the beamline
            # and has a valid trace space
            
            num_beamline_elements = 9
            x=[]
            y=[]
            ke=[]
            
            if len(trace_space) > num_beamline_elements and len(trace_space[num_beamline_elements]) >= 3:
                phase = Prtcl.Particle.RPLCTraceSpace2PhaseSpace(trace_space[-1])

                phase = np.array(phase)
                x_end, y_end = phase[0, 0], phase[0, 1]
                px, py, pz = phase[1, 0], phase[1, 1], phase[1, 2]
                if pz is not None:
                    ke_end = np.sqrt(px**2 + py**2 + pz**2 + pmass**2) - pmass
                    x.append(x_end)
                    y.append(y_end)
                    ke.append(ke_end)

        histogram_counts = cls.histo_data(x, y, ke)
        exp_data = 
        
        chi_squared_summed = cls.compare_histograms(histogram_counts, exp_data)
        
        

        print('*************************************************************')
        print(f"_________Total chi squared value_________: {chi_squared_summed:.2f}%")
        print('*************************************************************')
        return chi_squared_summed
    
    def chi_squared(cls, exp_count, sim_count):
        chisq_array = (exp_count - sim_count)**2 / (sim_count + 1e-10)
        chisq = np.sum(chisq_array)
        return chisq
    
    def compare_histograms(cls, sim_counts, exp_counts):

    
        for i in range(7):
            
            new_chisq = cls.chi_squared(exp_counts, sim_counts)

            if 'chisq' not in locals():
                chisq = new_chisq
            else:
                chisq = np.concatenate((chisq, new_chisq), axis=-1)
        

        chisq_tot = np.sum(chisq)

        return chisq_tot
    
    def histo_data(cls, x, y, ke):
    
        # Energy cut levels
        stack_fn_cut = [3., 5.821401202856385, 7.997094322613386, 9.658781341633036, 11.148286563497997, 12.435592394828737, 13.723555017136873] 
        # These are the min energies before a reponse in the reponse function

        print('Getting histogram data')

        for stack in range(len(stack_fn_cut)):

            # Filter particles based on energy cut
            cut_x = []
            cut_y = []
            cut_energies = []
            
            for i in range(len(ke)):
                if ke[i] > stack_fn_cut[stack]:
                    cut_energies.append(ke[i])
                    cut_x.append(x[i]*1000)
                    cut_y.append(y[i]*1000)
            
            deposited_energies = response_func(cut_energies)
            
            # 2D Histogram (Main plot)
            new_counts, xedges, yedges = np.histogram2d(cut_x, cut_y, weights=deposited_energies, bins=(50,50))
            new_counts_norm = new_counts / np.sum(new_counts)
            new_counts_expnd_dims = np.expand_dims(new_counts_norm, axis=-1)
            if 'counts' not in locals():
                counts = new_counts_expnd_dims
            else:
                counts = np.concatenate((counts, new_counts_expnd_dims), axis=-1)

        return counts

#--------  Exceptions:
class noReferenceParticle(Exception):
    pass

