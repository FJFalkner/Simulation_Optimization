
import numpy as np
import matplotlib.pyplot as plt

def explicit_Euler(dydt, t_span, y0, n):

    # step size
    h = (t_span[1]-t_span[0]) / n
    # time vector
    t = np.linspace(t_span[0], t_span[1], n+1)
    # initialize solution vector
    y = np.zeros((n+1, len(y0)))
    # intial conditions
    y[0,:] = y0

    # *** start integration ***
    for i in range(n):
        # explicit Euler
        y[i+1,:] = y[i,:] + h * dydt(t[i], y[i])

    return t, y

def midpoint_rule(dydt, t_span, y0, n):

    # step size
    h = (t_span[1]-t_span[0]) / n
    # time vector
    t = np.linspace(t_span[0], t_span[1], n+1)
    # initialize solution vector
    y = np.zeros((n+1, len(y0)))
    # intial conditions
    y[0,:] = y0

    # *** start integration ***
    for i in range(n):

        # tangents
        d1 = dydt(t[i], y[i,:])
        d2 = dydt(t[i] + h/2, y[i,:] + h/2 * d1)

        # midpoint rule
        y[i+1,:] = y[i,:] + h * d2

    return t, y

def Runge_Kutta_4(dydt, t_span, y0, n):
    
    # step size
    h = (t_span[1]-t_span[0]) / n
    # time vector
    t = np.linspace(t_span[0], t_span[1], n+1)
    # initialize solution vector
    y = np.zeros((n+1, len(y0)))
    # intial conditions
    y[0,:] = y0

    # *** start integration ***
    for i in range(n):

        # tangents
        d1 = dydt(t[i], y[i,:])
        d2 = dydt(t[i] + h/2, y[i,:] + h/2 * d1)
        d3 = dydt(t[i] + h/2, y[i,:] + h/2 * d2)
        d4 = dydt(t[i] + h, y[i,:] + h * d3)

        # Runge-Kutta 4
        y[i+1,:] = y[i,:] + h * (d1 + 2*d2 + 2*d3 + d4)/6

    return t, y


