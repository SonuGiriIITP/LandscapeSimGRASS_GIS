import numpy
from scipy import *

def CityBlock(image_arr,flag):
  """
  Args:
    image_arr: input array whose distance needs to be calculated (2-D array of int or flat)
    flag: positive number,if image_arr is a binary array else 0 for non-binary image that needs to be thresholded
  output:
    distance_arr: contains city-block distance values
  """
  (x_len,y_len)=image_arr.shape
  if flag == 0:
    # threshold the non-binary array
    L = []
    for i in range(0,x_len):
      for j in range(0,y_len):
        L.append((image_arr[i][j],i,j))
    L.sort()
    binary_arr = numpy.zeros((x_len, y_len),dtype="uint8") 
    for i in range(len(L)-1 , 95*len(L)/100,-1):
      binary_arr[ L[i][1] ][ L[i][2] ] = 255
  else:
      binary_arr = image_arr
  distance_arr=zeros((x_len,y_len), dtype= "int")
  for i in range(0, x_len):
    for j in range(0, y_len):
      if binary_arr[i][j]==0:
        distance_arr[i][j] = x_len + y_len+1
      else:
        distance_arr[i][j] = 0
  # Apply distance transform
  for i in range(0, x_len):
    for j in range(0, y_len):
      if i-1 < 0 and j-1 >=0:
        distance_arr[i][j] = min(distance_arr[i][j],1+distance_arr[i][j-1])
      elif i-1 >= 0 and j-1 < 0:
        distance_arr[i][j] = min(distance_arr[i][j],1+distance_arr[i-1][j])
      elif i-1 >= 0 and j-1 >= 0:
        distance_arr[i][j] = min(distance_arr[i][j],1+distance_arr[i-1][j], 1+distance_arr[i][j-1])
      else:
        distance_arr[i][j]=distance_arr[i][j]
  for i in range(x_len-1, -1, -1):
    for j in range(y_len-1, -1, -1):
      if i+1 > x_len-1 and j+1 <=y_len-1:
        distance_arr[i][j] = min(distance_arr[i][j],1+distance_arr[i][j+1])
      elif i+1 <= x_len-1 and j+1 > y_len-1:
        distance_arr[i][j] = min(distance_arr[i][j],1+distance_arr[i+1][j])
      elif i+1 <= x_len-1 and j+1 <= y_len-1:
        distance_arr[i][j] = min(distance_arr[i][j],1+distance_arr[i+1][j], 1+distance_arr[i][j+1])
      else:
        distance_arr[i][j]=distance_arr[i][j]
  return distance_arr
