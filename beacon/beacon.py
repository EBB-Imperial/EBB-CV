import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, Eq, solve

# Define the variables


import sympy as sp

def find_pos(alpha, beta, w, h, diff=0.5):
    beacon1 = (0, h)
    beacon2 = (w, h)
    beacon3 = (w, 0)
    # w = float(w)
    # h = float(h)


    alpha = sp.rad(alpha)
    beta = sp.rad(beta)
    list = []
    for x in range(0, w, 1):
        for y in range(0, h, 1):
            alpha1 = math.atan(x/(h-y)) + math.atan((w-x)/(h-y))
            beta1 = math.atan((h-y)/(w-x)) + math.atan((y/(w-x)))
            if alpha - diff < alpha1 < alpha + diff and beta - diff < beta1 < beta + diff:
                list.append([x, y])
    return list


def find_pos_1(alpha, beta, w, h, diff=0.5):
    beacon1 = (0, h)
    beacon2 = (w, h)
    beacon3 = (w, 0)
    # w = float(w)
    # h = float(h)
    
    x = sp.Symbol('x')

    equation = sp.Eq(10 - 2*x, 3*x)

    solution = sp.solve(equation, x)
  
    # return solution[x], solution[y]


def plot(width, height, list):
    origin_x, origin_y = 0, 0
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Draw the rectangle
    rectangle = plt.Rectangle((origin_x, origin_y), width, height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rectangle)

    # Draw the point on the rectangle as a red circle
    for element in list:
        [point_x, point_y] = element
        ax.plot(point_x, point_y, marker='o', markersize=1, color='red')

    # Set the axis limits based on the rectangle dimensions
    ax.set_xlim(origin_x, origin_x + width)
    ax.set_ylim(origin_y, origin_y + height)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Display the plot
    plt.show()

list = find_pos(120, 45, 100, 50, 0.01)

plot(100, 50, list)