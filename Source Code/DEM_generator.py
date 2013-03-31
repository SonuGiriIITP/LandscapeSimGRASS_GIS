#!/usr/bin/env python
import pylab
import numpy
import random
import math

def fN(delta, x, sigma):
    '''Takes an n length numpy numpy.array, x, and 'blurs' the mean
       using a Gaussian distrubition centered about 0. The
       Gaussian distribution is scaled by delta and has standard
       deviation of sigma.'''
    return_num = x.mean() + delta * random.gauss(0.0, sigma)
    return return_num

def midPointFm2d(max_level, sigma, H, addition, wrap, gradient,
                 seed=0, normalise = True, bounds = [0,1], gradient_values = [30,28,23,25]):
    """
    ________________________________________________________________________
    Args:
        max_level : Maximum number of recursions( N = 2^max_level)
        sigma     : Initial standard deviation
        H         : Roughness constant varies form 0.0 to 1.0
        addition  : boolean parameter (turns random additions on/off)
        wrap      : wraps the Image
        gradient  : if 1 then corners are deterministically set else randomally
        seed      : seed value for random number generator
        normalise : normalizes the data using bound
        bounds    : used for normalization of the grid data
        gradient_values: list of values for corner points of grid in case 
                         gradient is non-zero
    Result:     
        Output is given in the form of an array(grid) which holds surface
        elevation data for a square region. Write the elevation values in an
        ascii file.
    _________________________________________________________________________
    """	    
    random.seed(seed)            #seed the random number generator 
    north      = gradient_values[0]
    west       = gradient_values[1]
    south      = gradient_values[2]
    east       = gradient_values[3]

    N = 2**max_level 
    grid = numpy.zeros([N+1,N+1])#Generate a 2-D real array of size (N+1)^2 initialized to zero 
    delta = sigma                # delta is a real variable holding standard deviations
    if gradient == 1:            #set the initial random corners
        grid[0,0] = north 
        grid[0,N] = east
        grid[N,N] = south
        grid[N,0] = west
    else:
        grid[0,0] = delta*random.gauss(0.0,sigma) 
        grid[0,N] = delta*random.gauss(0.0,sigma)
        grid[N,N] = delta*random.gauss(0.0,sigma)
        grid[N,0] = delta*random.gauss(0.0,sigma)
    D = N
    dd = N/2
    def locFN(x):
        return fN(delta, x, sigma)

    for stage in numpy.arange(1,max_level+1):        
        delta = delta / 2**(H/2.0) #going from grid type I to grid Type II
        vec = range(dd, N-dd+1, D)
        #interpolate and offset points
        grid[dd:N-dd+1:D,dd:N-dd+1:D] = [[fN(delta, numpy.array([grid[x+dd,y+dd],\
			grid[x+dd,y-dd],grid[x-dd,y+dd], grid[x-dd,y-dd]]), sigma)\
            for y in vec] for x in vec]

        if addition:
            vec = range(0,N+1,D)
            grid[0:N+1:D,0:N+1:D] += [[delta*random.gauss(0.0,sigma)\
				for y in vec]for x in vec]

        delta = delta / 2**(H/2.0)   #going from grid type II to grid type I
        for x in range(dd,N-dd+1,D): #interpolate and offset boundary grid points
            grid[x,0] = fN(delta, numpy.array([grid[x+dd,0],grid[x-dd,0],\
				grid[x,dd]]),sigma)
            grid[x,N] = fN(delta, numpy.array([grid[x+dd,N],grid[x-dd,N],\
				grid[x,N-dd]]),sigma)
            grid[0,x] = fN(delta, numpy.array([grid[0,x+dd],grid[0,x-dd],\
				grid[dd,x]]),sigma)
            grid[N,x] = fN(delta, numpy.array([grid[N,x+dd],grid[N,x-dd],\
				grid[N-dd,x]]),sigma)
            if wrap:
                grid[x,N] = grid[x,0]
                grid[N,x] = grid[0,x]

        x_vec = range(dd,N-dd+1,D)
        y_vec = range(D,N-dd+1,D)
	#interpolate offset interior grid points
        grid[dd:N-dd+1:D,D:N-dd+1:D] = [[fN(delta, numpy.array([grid[x,y+dd],\
			grid[x,y-dd],grid[x+dd,y],grid[x-dd,y]]), sigma)\
			for y in y_vec] for x in x_vec]

        x_vec = range(D,N-dd+1,D)
        y_vec = range(dd,N-dd+1,D)

        if x_vec != []:
            grid[D:N-dd+1:D,dd:N-dd+1:D] = [[fN(delta,\
					numpy.array([grid[x,y+dd], grid[x,y-dd],\
                    grid[x+dd,y], grid[x-dd,y]]), sigma)\
					for y in y_vec] for x in x_vec]
        if addition:          
            vec = range(0,N+1,D)
            grid[0:N+1:D,0:N+1:D] += [[delta*random.gauss(0.0,sigma)\
				for y in vec]for x in vec]
            vec = range(dd,N-dd+1,D)
            grid[dd:N-dd+1:D,dd:N-dd+1:D] += [[delta*random.gauss(0.0,sigma)\
				for y in vec]for x in vec]
        D=D/2
        dd=dd/2
    if(normalise):
        grid += numpy.amin(grid)*-1
        grid = (grid/numpy.amax(grid)) * (bounds[1] -bounds[0])+ bounds[0]
    return grid


def SpectralSynthesisFM2D(max_level, sigma, H, seed=0, normalise=True, bounds=[0,1]):
    """
    ________________________________________________________________________
    Args:
        max_level : Maximum number of recursions( N = 2^max_level)
        sigma     : Initial standard deviation
        H         : Roughness constant varies form 0.0 to 1.0
        seed      : seed value for random number generator
        normalise : normalizes the data using bound
        bounds    : used for normalization of the grid data
    Result:     
        Output is given in the form of an array(grid) which holds surface
        elevation data for a square region.  
    _________________________________________________________________________
    """	
    N = 2**max_level 
    A = numpy.zeros((N,N), dtype = complex)
    random.seed(seed) #seed the random number generator
    PI = 3.141592
    for i in range(0,N/2):
        for j in range(0,N/2):
            phase = 2*PI*random.random()#/random.randrange(1,Arand)
            if i != 0 or j != 0:
                rad = pow((i*i + j*j),(-(H+1)/2) )*random.gauss(0.0, sigma)
            else:
                rad = 0.0
            A[i][j] = rad*math.cos(phase) + rad*math.sin(phase)*j 
            if i ==0: 
                i0 = 0
            else:
                i0 = N - i
            if j==0:
                j0 = 0
            else:
                j0 = N - j
    
            A[i0][j0] = rad * math.cos(phase) - rad*math.sin(phase)*j
  
    for i in range(1,N/2):
        for j in range(1,N/2):
            phase = 2*PI*random.random()#/random.randrange(1,Arand)
            rad = pow((i*i + j*j),(-(H+1)/2) )*random.gauss(0.0, sigma)
            A[i][N-j] = rad * math.cos(phase) + rad* math.sin(phase)*j
            A[N-i][j] = rad * math.cos(phase) - rad* math.sin(phase)*j
  
    grid = numpy.real(pylab.ifft2(( A ) ))
    if(normalise):
        grid += numpy.amin(grid)*-1
        grid = (grid/numpy.amax(grid)) * (bounds[1] -bounds[0])+ bounds[0]
    return grid

def DEM_creator(H, H_wt, seed, elev_range,sigma,gradient, max_level, DEMcreator_option):
    """
    Generates a DEM map with specified parameters described below.
    Args:
        H    : List containing H values         --> [H1, H2, H3, ...]( List of floats )
        H_wt : List containing H_wt             --> [H1_wt, H2_wt, H3_wt, ...] ( List of floats )
        seed : seed for random no generator     --> [seed1, seed2 seed3, ...] ( list of ints)
        elev_range: List containing elev bounds --> [elev_min, elev_max] (int, int)
        sigma: Initial standard deviation
        gradient: if 1 then corners are deterministically set else randomally
        max_level : size of grid 2^(max_level) + 1   ( int )
        DEMcreator_option: Specify the method used to generate DEM (fm2D or SS)       
    Result:
        Output : [ DEM_arr, [List of DEM input grids], [ List of DEM input grid names ], [list of parameters for DEM input grid]]
               where DEM_arr :Final Output DEM grid  
    """
    Input_DEMs = []
    name = []
    parameter = []
    DEM_arr = numpy.zeros((2**max_level,2**max_level),dtype=float)

    if DEMcreator_option == 'fm2D':       
        for i in range(0,len(H)):    
            #Generate other DEM's with gradient = 0 (i.e. FLASE) and method specified by DEMcreator_option         
            temp_arr = midPointFm2d(max_level = max_level, sigma = sigma, H = H[i], addition = True,\
                                     wrap = False, gradient = 0, seed = seed[i],normalise = True,
                                     bounds = elev_range,gradient_values = [30,28,23,25])
            #TODO gradient ignored for now
            Input_DEMs.append(temp_arr)
            file_name = "InputDEM_arr%d" % (i+1)
            name.append(file_name)
            parameter.append((H[i], H_wt[i]))
            DEM_arr = DEM_arr +  H_wt[i]*temp_arr[:-1,:-1]
    else:
        for i in range(0,len(H)):
            #Generate other DEM's with gradient = 0 (i.e. FLASE) and method specified by DEMcreator_option i.e SS in this case    
            temp_arr = SpectralSynthesisFM2D(max_level = max_level, sigma = sigma, H = H[i],\
                                                   seed = seed[i], normalise = True, bounds = elev_range)
            Input_DEMs.append(temp_arr)
            file_name = "InputDEM_arr%d" % (i+1)
            name.append(file_name)
            parameter.append((H[i], H_wt[i]))
            DEM_arr = DEM_arr +  H_wt[i]*temp_arr

    Output = [DEM_arr, Input_DEMs, name, parameter] 
    return Output

#For testing purpose
#if __name__ == "__main__":
#    arr = DEM_creator(H=[0.7,0.8,0.9], H_wt=[0.3,0.4,0.2], seed=[5,7,31], elev_range=[200,1000],sigma=1,gradient=0, max_level=9, DEMcreator_option='SS')
#    print arr[0], 'works okk'
