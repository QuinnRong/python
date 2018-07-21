import math
import numpy as np

def cross(a, b):
    '''
    equivilent np.cross(np.array(a), np.array(b))
    '''
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]
    return c

def intersection(ang1, d1, ang2, d2):
    '''
    input: angles and distances of two lines
    line functions: 
    x*cos(theta1) + y*sin(theta1) - d1 = 0;
    x*cos(theta2) + y*sin(theta2) - d2 = 0;
    output: intersection of two lines
    '''
    theta1 = ang1/180*math.pi
    theta2 = ang2/180*math.pi
    a = [math.cos(theta1), math.sin(theta1), -d1]
    b = [math.cos(theta2), math.sin(theta2), -d2]
    c = cross(a, b)         
    print("intersection is: (x=%f, y=%f)" % (c[0]/c[2], c[1]/c[2]))

def points2line(x1, y1, x2, y2):
    '''
    input: coordinates of two points
    output: angle and distance of the line crossing the two points
    '''
    a = [x1, y1, 1]
    b = [x2, y2, 1]
    c = cross(a, b)
    ang = math.acos(-np.sign(c[2])*c[0]/math.sqrt(c[0]**2 + c[1]**2))/math.pi*180
    d = abs(c[2])/math.sqrt(c[0]**2 + c[1]**2)
    print("line is: (ang=%.2fdeg, d=%f)" % (ang, d))

def main():
    intersection(30, 1, 60, 1)
    points2line(-2, 1, -1, 2)

if __name__ == '__main__':
    main()