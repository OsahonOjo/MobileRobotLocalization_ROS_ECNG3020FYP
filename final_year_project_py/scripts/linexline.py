import math
import matplotlib.pyplot as plt

def linexline(L1x, L1y, L2x, L2y, showIntersectionPlot=True):
    # nargin check
    if len(L1x) != 2 or len(L1y) != 2 or len(L2x) != 2 or len(L2y) != 2:
        raise ValueError('Invalid input arguments.')
    
    if len(L1x) != len(L1y) != len(L2x) != len(L2y):
        raise ValueError('Invalid input arguments.')
    
    # Show intersection plot
    if len(L1x) == len(L1y) == len(L2x) == len(L2y) == 4:
        showIntersectionPlot = True

    # Data
    x1 = L1x[0]
    y1 = L1y[0]
    x2 = L1x[1]
    y2 = L1y[1]
    x3 = L2x[0]
    y3 = L2y[0]
    x4 = L2x[1]
    y4 = L2y[1]

    # MATLAB behavior
    # >> 0/0 = NaN
    # >> 1/0 = Inf
    # >> -1/0 = -Inf

    # Line segments intersect parameters
    try:
        u_numerator = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2))
        u_denominator = ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        u = u_numerator / u_denominator
    except ZeroDivisionError:
        if (u_numerator > 0):
            u = float('inf')
        elif(u_numerator == 0):
            u = math.nan
        elif(u_numerator < 0):
            u = float('-inf')

    try:
        t_numerator = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))
        t_denominator = ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        t = t_numerator / t_denominator
    except ZeroDivisionError:
        if (t_numerator > 0):
            t = float('inf')
        elif(t_numerator == 0):
            t = math.nan
        elif(t_numerator < 0):
            t = float('-inf')


    # Check if intersection exists, if so then store the value
    xi = ((x3 + u * (x4-x3)) + (x1 + t * (x2-x1))) / 2
    yi = ((y3 + u * (y4-y3)) + (y1 + t * (y2-y1))) / 2

    if (u >= 0 and u <= 1.0) and (t >= 0 and t <= 1.0):
        real = 1
    else:
        real = 0

    if showIntersectionPlot:
        # Plot the lines
        plt.plot([x1, x2], [y1, y2], linewidth=3)
        plt.plot([x3, x4], [y3, y4], linewidth=3)

        # Plot intersection points
        plt.plot(x3 + u * (x4-x3), y3 + u * (y4-y3), 'ro', markersize=15)
        plt.plot(x1 + t * (x2-x1), y1 + t * (y2-y1), 'bo', markersize=15)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.savefig('LineXPlot.png')

    return xi, yi, real


# Test cases

# Test case 1
L1x = [0, 4]
L1y = [0, 0]
L2x = [2, 2]
L2y = [1, -1]
xi_expected = 2.0
yi_expected = 0.0
real_expected = 1

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 1: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 1: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 1: Expected real: {real_expected}, got: {real}")
print()

# Test case 2
L1x = [0, 4]
L1y = [0, 0]
L2x = [2, 5]
L2y = [1, 1]
xi_expected = float('-inf')
yi_expected = math.nan
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 2: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 2: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 2: Expected real: {real_expected}, got: {real}")
print()

# Test case 3
L1x = [-1, 1]
L1y = [-1, 1]
L2x = [1, -1]
L2y = [-1, 1]
xi_expected = 0.0
yi_expected = 0.0
real_expected = 1

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 3: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 3: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 3: Expected real: {real_expected}, got: {real}")
print()

# Test case 4
L1x = [0, 2]
L1y = [0, 2]
L2x = [2, 4]
L2y = [2, 0]
xi_expected = 2.0
yi_expected = 2.0
real_expected = 1

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 4: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 4: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 4: Expected real: {real_expected}, got: {real}")
print()

# Test case 5
L1x = [0, 2]
L1y = [0, 2]
L2x = [1, 3]
L2y = [1, 3]
xi_expected = math.nan
yi_expected = math.nan
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 5: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 5: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 5: Expected real: {real_expected}, got: {real}")
print()

# Test case 6
L1x = [0, 0]
L1y = [0, 2]
L2x = [1, 1]
L2y = [1, 3]
xi_expected = math.nan
yi_expected = float('inf')
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 6: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 6: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 6: Expected real: {real_expected}, got: {real}")
print()

# Test case 7
L1x = [0, 4]
L1y = [0, 0]
L2x = [0, 2]
L2y = [1, 1]
xi_expected = float('-inf')
yi_expected = math.nan
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 7: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 7: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 7: Expected real: {real_expected}, got: {real}")
print()

# Test case 8
L1x = [0, 4]
L1y = [0, 0]
L2x = [1, 3]
L2y = [1, 1]
xi_expected = float('-inf')
yi_expected = math.nan
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 8: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 8: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 8: Expected real: {real_expected}, got: {real}")
print()

# Test case 9
L1x = [0, 4]
L1y = [0, 0]
L2x = [2, 5]
L2y = [1, 1]
xi_expected = float('-inf')
yi_expected = math.nan
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 9: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 9: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 9: Expected real: {real_expected}, got: {real}")
print()

# Test case 10
L1x = [0, 0]
L1y = [0, 2]
L2x = [1, 1]
L2y = [1, 3]
xi_expected = math.nan
yi_expected = math.nan
real_expected = 0

xi, yi, real = linexline(L1x, L1y, L2x, L2y)

print(f"Test case 10: Expected xi: {xi_expected}, got: {xi}")
print(f"Test case 10: Expected yi: {yi_expected}, got: {yi}")
print(f"Test case 10: Expected real: {real_expected}, got: {real}")
print()