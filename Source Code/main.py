#!/usr/bin/env python
import yaml
import DEM_generator
import dist
import Geometry
import rpy
import grass.script as g
import pylab
import numpy
import sys
import os
import time
from scipy import ndimage


def Erosion(River_arr, DEM_arr, river_drop):
    """
    Carry out erosion given the DEM and flow accumulation matrix
    Args:
        River_arr : 2-D array containing flow accumulation values ( 2-D array of ints )
        DEM_arr   : DEM array (an array of floats)
        river_drop: describes the maximum extent of erosion to be performed
                    A proportion of this value is subtracted from pixel depending on distance from the river (float)
    Result:
        DEM_arr: Eroded digital elevation model(DEM) array ( 2-D array of floats )
    """
    (x_len, y_len) = DEM_arr.shape
    # Get the distance map from flow accumulation matrix i.e. River_arr
    Distance_arr = dist.CityBlock(River_arr,flag = 0)
    # Create a mask for differnet distances used for DEM erosion
    mask7 = [Distance_arr > 25]
    mask6 = [Distance_arr < 50]
    mask4 = [Distance_arr <= 25]
    mask5 = [Distance_arr > 3]
    mask3 = [Distance_arr == 3]
    mask2 = [Distance_arr == 2]
    mask1 = [Distance_arr == 1]
    mask0 = [Distance_arr == 0]
    max_flow_accum = numpy.max(River_arr)
    for i in range(0,x_len):
        for j in range(0,y_len):
            #Erode the landscape based on it's distance from river
            if mask0[0][i][j] == True:
                DEM_arr[i][j] = DEM_arr[i][j] - river_drop
            elif mask1[0][i][j] == True:
                DEM_arr[i][j] = DEM_arr[i][j] - (river_drop * 0.9)
            elif mask2[0][i][j] == True:
                DEM_arr[i][j] = DEM_arr[i][j] - (river_drop * 0.8)
            elif mask3[0][i][j] == True:
                DEM_arr[i][j] = DEM_arr[i][j] - (river_drop * 0.7)
            elif mask4[0][i][j] == True and mask5[0][i][j] == True:
                DEM_arr[i][j] = DEM_arr[i][j] - (river_drop * 0.55)
            elif mask6[0][i][j] == True and mask7[0][i][j] == True:
                DEM_arr[i][j] = DEM_arr[i][j] - (river_drop * 0.4)
            else:
                DEM_arr[i][j] = DEM_arr[i][j] - (river_drop * 0.1)
    return DEM_arr


def DEM_arr_to_ascii(DEM_arr,resolution):
    """
    Given a DEM array DEM_arr_to_ascii write it into an ascii file
    with suitable header
    Input:
        DEM_arr: Digital elevation array (2D array of float)
        resolution: DEM cell resolution (int)
    """
    # header info
    fo = open("ascii_files/DEM.asc","w")
    north = 4928050 #some arbitrary value
    south = north - resolution*DEM_arr.shape[0]
    east =  609000  #some arbitrary value
    west =  east - resolution*DEM_arr.shape[1]
    rows = DEM_arr.shape[0]
    cols = DEM_arr.shape[1]
    null = -9999
    fo.write("north: " + str(north)+"\n")
    fo.write("south: " + str(south)+"\n")
    fo.write("east: "  + str(east) +"\n")
    fo.write("west: "  + str(west) +"\n")
    fo.write("rows: "  + str(rows) +"\n")      
    fo.write("cols: "  + str(cols) +"\n")
    fo.write("null: "  + str(null) +"\n")
    for i in range(0,DEM_arr.shape[0]):
        fo.write("\n")
        for j in range(0,DEM_arr.shape[1]):
            fo.write(str(DEM_arr[i][j])+ " ")
    fo.close()


def DecisionTree(no_of_veg_class, elev_filename, landcover_filename, river_filename):
    """
    Generates a decision tree given the training data
    Input:
        no_of_veg_class: No of landcover class in training data
        elev_filename  : Name of training file having elevation values
        landcover_filename: Name of training file having landcover values
        river_filename: Name of training file having river presence absence info
    """
    rpy.r.library("rpart")
    g.use_temp_region()
    #TODO generalize no of rows and columns for training data
    g.run_command('g.region', flags = 'ap',res = 50, rows = 2001 ,cols = 1201)
    pathname = os.path.dirname(sys.argv[0])        
    fullpath = os.path.abspath(pathname)
    # Convert ascii DEM into grass raster map that will help in getting slope and aspect
    file_name = "/Training/%s" % elev_filename
    g.run_command('r.in.ascii', overwrite = True, flags='f', input = fullpath + file_name, output='training_DEM')
    # TODO read training DEM into array without writing another file 
    g.run_command('r.out.ascii',flags='h',input='training_DEM@user1',output=fullpath + '/ascii_files'+'/training_DEM',null='0')
    f = open('ascii_files/training_DEM', 'r')
    Elev_arr = numpy.loadtxt(f,unpack = True)
    f.close() 
    file_name = "Training/%s" % (landcover_filename)
    Landcover = numpy.loadtxt(file_name, unpack=True) # Read Landcover Data from ascii file
    file_name = "Training/%s" % (river_filename)
    River     = numpy.loadtxt(file_name, unpack=True) # Read River Data from ascii file
    River_dist_arr = dist.CityBlock(River,flag = 1)   #Compute distance from River data
    g.run_command('r.slope.aspect',overwrite=True,elevation='training_DEM@user1',slope='Slope',aspect='Aspect')
    g.run_command('r.out.ascii',flags='h',input='Slope@user1',output=fullpath + '/ascii_files'+'/Slope',null='0')
    f = open('ascii_files/Slope', 'r')
    Slope_arr = numpy.loadtxt(f,unpack = True)  #Get Slope into an array
    f.close()
    g.run_command('r.out.ascii',flags='h',input='Aspect@user1',output=fullpath +'/ascii_files'+ '/Aspect',null='0')
    f = open('ascii_files/Aspect', 'r')
    Aspect_arr = numpy.loadtxt(f,unpack = True) #Get Aspect into an array
    f.close()

    (x_len,y_len) = Elev_arr.shape
    L = [ [] for i in range(0,no_of_veg_class)]
    for i in range(1,x_len-1):   # Ignoring boundary cells 
        for j in range(1,y_len-1):
            # Append the pixel co-ordinates into respective list of lists
            # nodata values already gets handled since we are ignoring it
            for k in range(0, no_of_veg_class):
                if Landcover[i][j] == k:
                    L[k].append( (i,j) )
                    break

    minimum_elev = numpy.min(Elev_arr)
    factor = numpy.max(Elev_arr) - minimum_elev      # normalize elevation data
    Elev_arr = (Elev_arr[:,:]-minimum_elev)*100/factor
    # Sample training Data for decision tree, we can't take entire data as it take longer processing time
    # various lists to hold sample training data
    Elevation = []
    Slope = []
    RiverDistance = []
    Aspect = []
    Class = []
    # Sample the data
    for i in range(0,no_of_veg_class):  
        if len(L[i]) < 500:
            limit = len(L[i])
        else:
            limit = 500
        for j in range(0,limit):
            Elevation.append( int(Elev_arr[ L[i][j][0] ][ L[i][j][1] ]))
            Slope.append(int(Slope_arr[ L[i][j][0] ][ L[i][j][1] ]))
            RiverDistance.append(int(River_dist_arr[ L[i][j][0] ][ L[i][j][1] ]))
            Aspect.append(int(Aspect_arr[ L[i][j][0] ][ L[i][j][1] ]))
            Class.append(i)

    # create dictionary of sample data which will be needed to generate decision tree 
    traing_data = {'Elevation':Elevation,'Slope':Slope,'RiverDistance':RiverDistance,'Aspect':Aspect,'Class':Class}
    rpy.set_default_mode(rpy.NO_CONVERSION)
    #Using rpart create the decision tree
    fit = rpy.r.rpart(formula='Class ~ Elevation + RiverDistance + Slope + Aspect',data=traing_data,method="class")
    rpy.r.png("DecisionTree.png")  # Output a png image of the decision tree
    rpy.r.plot(fit)
    rpy.r.text(fit)
    rpy.r.dev_off()
    return fit


def main():
    """
    It then performs the following:
    1. Gets all the parameters required for simulation from parameter.yaml file. 
    2. calls DEM_creator() --> for generating DEM grid
    3. Erosion modelling 
    4. Flow modelling
    5. Landcover class allocation using decision tree
    """
    time1 = time.time()
    #*****************parameter handling *************************************
    # Get the parameters from parameter.yaml file
    yaml_file  = open('parameter.yaml', 'r')
    stream     = yaml.load(yaml_file)
    resolution = stream['resolution']
    H          = stream['H']
    H_wt       =  stream['H_wt']
    seed       = stream['seed']
    elev_range = stream['elev_range']
    max_level  = stream['max_level']
    DEMcreator_option = stream['DEMcreator_option']
    output_dir = stream['output_dir']
    river_drop = stream['river_drop']
    counter    = stream['counter']
    elev_filename      = stream['training_elev_filename']
    landcover_filename = stream['training_landcover_filename']
    river_filename     = stream['training_river_filename']
    no_of_veg_class    = stream['no_of_veg_class']
    min_area     = stream['min_area']
    max_area     = stream['max_area']
    aspect_ratio = stream['aspect_ratio']
    agri_area_limit = stream['agri_area_limit']
    yaml_file.close()
    #**************************print statistics***********************************
    print ("Running simulation with follwing parameters")
    print ("H: %s" % H)
    print ("H_wt: %s" % H_wt)
    print ("seed: %s" % seed)
    print ("elev_range: %s" % elev_range)
    print ("max_level: %s" % max_level)
    print ("DEMcreator_option: %s" % DEMcreator_option)
    print ("output_dir: %s" % output_dir)
    print ("River drop: %d" % river_drop)
    print ("counter: %d" % counter)
    print ("no of vegetation class %d" % no_of_veg_class)
    print ("min area: %f" % min_area)
    print ("max area: %f" % max_area)
    print ("aspect ratio: %f" % aspect_ratio)
    print ("agricultural area limit: %f" % agri_area_limit)
    sigma = 1      #fixed
    gradient = 0   #fixed
    #*****************************DEM genaration************************************
    # Generate DEM using FM2D/SS algorithm by calling DEM_creator(args...) function
    DEM_Result = DEM_generator.DEM_creator(H, H_wt, seed, elev_range,sigma,gradient,max_level, DEMcreator_option)
    file_name = "%s/Original_DEM" % (output_dir)
    pylab.imsave(file_name, DEM_Result[0])
    for i in range(0,len(DEM_Result[1])):
        file_name = "%s/%sH_%fHwt_%f" % (output_dir,DEM_Result[2][i],DEM_Result[3][i][0],DEM_Result[3][i][1])
        file_name = file_name.replace('.','')
        pylab.imsave(file_name, DEM_Result[1][i])

    DEM_arr = DEM_Result[0]
    #****************************region adjustment***********************************
    # We create a temporary region that is only valid in this python session
    g.use_temp_region()
    rows = DEM_arr.shape[0]
    cols = DEM_arr.shape[1]
    n = 4928050 #some arbitrary value
    s = n - resolution*rows
    e = 609000  #some arbitrary value
    w = e - resolution*cols
    g.run_command('g.region', flags = 'ap', n = n ,s = s, e = e, w = w,res = resolution, rows = rows ,cols = cols)
   
    #*************************Flow accumulation with Erosion modelling****************************
    pathname = os.path.dirname(sys.argv[0])
    fullpath = os.path.abspath(pathname)
    for iteration in range(0,counter):
        DEM_arr_to_ascii(DEM_arr,resolution)
        #Input the DEM ascii file into grass
        g.run_command('r.in.ascii', overwrite = True, flags='f', input = fullpath +'/'+'ascii_files' +'/DEM.asc', output='test_DEM')
        g.run_command('r.out.png', input='test_DEM@user1', output = fullpath + '/'+output_dir +'/'+'DEM'+str(iteration))
        #Flow computation for massive grids (float version) 
        g.run_command('r.terraflow', overwrite = True, elevation = 'test_DEM@user1', filled = 'flooded_DEM',\
          direction = 'DEM_flow_direction',swatershed = 'DEM_sink_watershed', accumulation = 'DEM_flow_accum', tci = 'DEM_tci')
        g.run_command('r.out.png', input='flooded_DEM@user1', output = fullpath + '/'+output_dir +'/'+'flooded_DEM'+str(iteration))
        g.run_command('r.out.png', input='DEM_flow_direction@user1', output = fullpath + '/'+output_dir +'/'+'flow_direction'+str(iteration))
        g.run_command('r.out.png', input='DEM_sink_watershed@user1', output = fullpath + '/'+output_dir +'/'+'watershed'+str(iteration))
        g.run_command('r.out.png', input='DEM_flow_accum@user1', output = fullpath +'/'+ output_dir +'/'+'flow_accumulation'+str(iteration))
        g.run_command('r.out.png', input='DEM_tci@user1', output = fullpath + '/'+output_dir +'/'+'tci'+str(iteration))
        g.run_command('r.out.ascii',flags='h',input='DEM_tci@user1',output=fullpath +'/ascii_files'+ '/DEM_flow_accum',null='0')
        f = open(fullpath +'/ascii_files'+ '/DEM_flow_accum', 'r')
        Flow_accum_arr = numpy.loadtxt(f,unpack = True)
        f.close()
        #call erosion modelling function
        DEM_arr = Erosion(Flow_accum_arr, DEM_arr, river_drop)
    #****************************landcover allocation using decision tree********************************
    # Get slope and Aspect using grass functions
    g.run_command('r.slope.aspect',overwrite=True,elevation='test_DEM@user1',slope='DEM_Slope',aspect='DEM_Aspect')
    g.run_command('r.out.png', input='DEM_Slope@user1', output = fullpath + '/'+output_dir+'/'+'Slope')
    g.run_command('r.out.ascii',flags='h',input='DEM_Slope@user1',output=fullpath + '/ascii_files'+'/DEM_Slope',null='0')
    f = open('ascii_files/DEM_Slope', 'r')
    DEM_Slope_arr = numpy.loadtxt(f,unpack = True)
    f.close()
    g.run_command('r.out.png', input='DEM_Aspect@user1', output = fullpath +'/'+ output_dir +'/'+'Aspect')
    g.run_command('r.out.ascii',flags='h',input='DEM_Aspect@user1',output=fullpath +'/ascii_files'+'/DEM_Aspect',null='0')
    f = open('ascii_files/DEM_Aspect', 'r')
    DEM_Aspect_arr = numpy.loadtxt(f,unpack = True)
    f.close()
    Distance_arr = dist.CityBlock(Flow_accum_arr,flag = 0)
    # Normalize the elevation values to use decision tree
    minimum_elev = numpy.min(DEM_arr)
    factor = numpy.max(DEM_arr) - minimum_elev
    Elev_arr = (DEM_arr[:,:] - minimum_elev)*100/factor
    # Create various list to hold test data
    Elevation = []
    Slope = []
    RiverDistance = []
    Aspect = []
    # Append the data into respective list
    for i in range(0,DEM_arr.shape[0]):
        for j in range(0,DEM_arr.shape[1]):
            Elevation.append(int(Elev_arr[i][j]))
            Slope.append(int(DEM_Slope_arr[i][j]))
            RiverDistance.append(int(Distance_arr[i][j]))
            Aspect.append(int(DEM_Aspect_arr[i][j]))
    # Create dictionary to apply R's predict command on it 
    Test_data = {'Elevation':Elevation ,'Slope':Slope ,'RiverDistance':RiverDistance,'Aspect':Aspect}
    # create decision tree from training data
    fit = DecisionTree(no_of_veg_class,elev_filename, landcover_filename, river_filename)
    g.run_command('g.region', flags = 'ap', n = n ,s = s, e = e, w = w,res = resolution, rows = rows ,cols = cols)
    # Alloctae vegetation array for holding predicted landcover values
    Veg_arr = numpy.zeros(DEM_arr.shape, dtype = "uint8")
    rpy.r.library("rpart")
    rpy.set_default_mode(rpy.BASIC_CONVERSION)
    # values contain probability values of the predicted landcover classes
    values = rpy.r.predict(fit,newdata=Test_data,method="class")
    for i in range(0,Veg_arr.shape[0]):
        for j in range(0,Veg_arr.shape[1]):
        # Get the class having max probability for each test data point
            a = ndimage.maximum_position(values[i*Veg_arr.shape[0] + j])
            Veg_arr[i,j] = (a[0]*25) # Assign them some value to facilitate visualization
    pylab.imsave(output_dir+"/landcover.png",Veg_arr)
    #*************************** Assigning geometric features************************
    # Allocate and initialize Suitabilty map 
    Suitability = numpy.zeros( DEM_arr.shape, dtype = "uint8")
    for i in range(0,DEM_arr.shape[0]):
        for j in range(0,DEM_arr.shape[1]):
            #TODO can use mask here, needs to be generalised
            if Veg_arr[i][j] == 0: # Ignore
                Suitability[i][j] = 0 
            elif Veg_arr[i][j] == 25: # Deciduous woodland
                Suitability[i][j] = 60 
            elif Veg_arr[i][j] == 50: # Coniferous woodland
                Suitability[i][j] = 55 
            elif Veg_arr[i][j] == 75: # Agriculture including pasture
                Suitability[i][j] = 98 
            elif Veg_arr[i][j] == 100: # Semi-natural grassland
                Suitability[i][j] = 90 
            elif Veg_arr[i][j] == 125: # Bog and swamp
                Suitability[i][j] = 50
            elif Veg_arr[i][j] == 150: # Heath
                Suitability[i][j] = 75 
            elif Veg_arr[i][j] == 175: # Montane habitat
                Suitability[i][j] = 20 
            elif Veg_arr[i][j] == 200: # Rock and quarry
                Suitability[i][j] = 30 
            elif Veg_arr[i][j] == 225: # Urban
                Suitability[i][j] = 80
    Display_fields = Geometry.GeometricFeature(Suitability, min_area,max_area ,aspect_ratio ,agri_area_limit)
    pylab.imsave(output_dir+"/fields.png",Display_fields)
    time2 = time.time()
    print "time taken", time2-time1

if __name__ == "__main__":
    main()
