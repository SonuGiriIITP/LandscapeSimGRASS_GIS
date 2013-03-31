import numpy
import random
from scipy import ndimage
import pylab

def GeometricFeature(Suitability, min_area = 40,max_area = 400,aspect_ratio = 1.8,agri_area_limit = 0.3):
  """
  Args
    Suitability: 2-D array containing suitability matrix
    min_area:      minimum area of a filed  (integer)
    max_area:      maximum area of a filed  (integer)
    aspect_ratio:  ratio of lenght to width ( float )
    agri_area_limit: ratio of area of agriculture to entire grid (0 < float < 1)
  """
  ID = 0  
  (x_len, y_len) = Suitability.shape
  Agri_arr = numpy.zeros((x_len,y_len),dtype = "int")
  Display_fields = numpy.zeros((x_len,y_len),dtype = "uint8")

  # Threshold the suitabilty map
  # TODO option for threshold in parameter list
  List = []
  for i in range(0,x_len):
    for j in range(0,y_len):
      List.append((Suitability[i][j],i,j))
  List.sort(reverse = True)
  threshold = List[int(0.7*len(List))][0]

  # Calculate the dimension of field using max_area and aspect_ratio and generate a rectangle
  height = numpy.sqrt(max_area/aspect_ratio)
  rectangle = numpy.zeros((int(height),int(aspect_ratio*height)),dtype="int")

  list_index = 0  #denote the suitability list element in consideration
  Covered_area = 0
  # limit1 denote the constraint on the area to be covered by agriculture
  limit1 = int(agri_area_limit*(x_len-1)*(y_len-1))
  angle_2 = int(random.random()*180)   # generate a random angle between 0 to 90
  #_____________________________________________________________________
  Patch_area_list = [0]
  while Covered_area < limit1 or list_index > len(List) - min_area:
    ID = ID + 1 # keep increasing the ID before every field placement
    rectangle[:,:] = ID
    # Introducing next patch orientation probability
    angle_1 = angle_2
    angle_2 = int(random.random()*180)
    if random.random() > 0.5:
      angle = angle_2
    else:
      angle = angle_1
      angle_2 = angle_1
    patch = ndimage.interpolation.rotate(rectangle,angle,axes=(1,0),reshape=True,\
                 output=None, order=3, mode='constant', cval=0.0, prefilter=True)
    # get the approximate centroid of the patch (rotated field) 
    (x,y) = (int((patch.shape[0])/2),int((patch.shape[1])/2))
    # get the next best pixel for placing field onto map
    (p,q) = (List[list_index][1], List[list_index][2])
    Area = 0 # needed to keep track of area of the field
    Buffer = [] # needed in case we have to discard the field
    for i in range(0,patch.shape[0]):
      for j in range(0,patch.shape[1]):
        (a,b) = (p-x+i,q-y+j) # get the absolute location in the map 
        if ( (a >= 1) and (b >= 1) and ( a <= x_len - 2) and (b <= x_len-2) ):
          """Check weather the pixel lie within the boundary of the grid"""
          # check condition for overlap ,inter-field distance, suitability
          if patch[i][j] > 0 and Agri_arr[a][b] == 0 and Suitability[a][b] >= threshold:
            Area = Area + 1
            Buffer.append((a,b))
            #TODO check inter-field distance

    if Area >= min_area:
      # place the field onto map
      Patch_area_list.append(Area)
      Covered_area = Covered_area + Area # increase total covered area
      for i in range(0,len(Buffer)):
        Agri_arr[Buffer[i]] = ID
        Display_fields[Buffer[i]] = 255
        List.remove((Suitability[Buffer[i]],Buffer[i][0],Buffer[i][1]))

      # mark the boundary of the field
      for i in range(0,len(Buffer)):
        (a,b) = Buffer[i]
        for i in range(-1,2):
          for j in range(-1,2):
            if Agri_arr[a-i][b-j] == 0:
              Agri_arr[a-i][b-j] = -1 #inter-patch strip indicator
              Display_fields[a-i][b-j] = 150
              List.remove((Suitability[a-i][b-j],a-i,b-j))
              Covered_area = Covered_area + 1
    else:
      list_index = list_index + 1
      ID = ID - 1
  #___________________________________________________________
  #           Create adjacency matrix for patch
  max_ID = numpy.max(Agri_arr)
  mask = [Agri_arr == -1]
  adjacency_matrix = numpy.zeros((max_ID+1,max_ID+1),dtype = bool)
  (x_len,y_len) = Agri_arr.shape
  for i in range(1,x_len-1):
    for j in range(1,y_len-1):
     if mask[0][i][j] == True:
       if Agri_arr[i-1][j] != Agri_arr[i+1][j]: 
         adjacency_matrix[ Agri_arr[i-1][j], Agri_arr[i+1][j] ] = True
         adjacency_matrix[ Agri_arr[i+1][j], Agri_arr[i-1][j] ] = True
       if Agri_arr[i][j-1] != Agri_arr[i][j+1]: 
         adjacency_matrix[ Agri_arr[i][j-1], Agri_arr[i][j+1] ] = True
         adjacency_matrix[ Agri_arr[i][j+1], Agri_arr[i][j-1] ] = True
       if Agri_arr[i-1][j-1] != Agri_arr[i+1][j+1]: 
         adjacency_matrix[ Agri_arr[i-1][j-1], Agri_arr[i+1][j+1] ] = True
         adjacency_matrix[ Agri_arr[i+1][j+1], Agri_arr[i-1][j-1] ] = True
       if Agri_arr[i+1][j-1] != Agri_arr[i-1][j+1]: 
         adjacency_matrix[ Agri_arr[i+1][j-1], Agri_arr[i-1][j+1] ] = True
         adjacency_matrix[ Agri_arr[i-1][j+1], Agri_arr[i+1][j-1] ] = True
  pylab.imsave("matrix.png",adjacency_matrix,cmap="gray")
  return Display_fields
