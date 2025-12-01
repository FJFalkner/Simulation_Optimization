
import numpy as np

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

def Euler_Richardson(dydt, t_span, y0, h0, TOL):
    
    # Unpack time interval
    t0 = t_span[0]
    t_end = t_span[1]

    y0 = np.array(y0, dtype=float)
    
    # Lists to store time and solution values
    t_list = [t0]
    y_list = [y0]
    
    h = h0
    
    while t_list[-1] < t_end:
        t = t_list[-1]
        y = y_list[-1]
        
        # Ensure we do not step beyond t_end
        if t + h > t_end:
            h = t_end - t
        
        # --- full step ---
        u = y + h * dydt(t, y)
        
        # --- half step ---
        v = y + (h/2) * dydt(t, y)
        w = v + (h/2) * dydt(t + h/2, v)
        
        # --- solution at next step ---
        y_new = 2*w - u
        
        # --- error estimation ---
        EST = np.max(np.abs(w - u))
        
        # --- adapt step size ---
        # Protect against division by zero
        if EST == 0:
            factor = 2
        else:
            factor = 0.9 * np.sqrt(TOL / EST)
        
        factor = min(2, max(0.2, factor))  # limit factor
        h_new = h * factor
        
        # --- store next step ---
        t_next = t + h
        t_list.append(t_next)
        y_list.append(y_new)
        
        h = h_new

    return np.array(t_list), np.vstack(y_list)


