import numpy as np
import testFunction as t

def initSimplex(bounds):
    l = bounds[0]
    u = bounds[1]
    return np.random.uniform(low = l, high = u, size = (3,2))

def minimize(func, max_iter):
    pt = []
    opt = []
    for i in range(10):
        simplex = initSimplex([-5,5])
        for i in range(max_iter):
            f = np.apply_along_axis(func, 1, simplex)
            idx = np.argsort(f)
            x0 = np.mean(np.vstack((simplex[idx[0]], simplex[idx[1]])), axis = 0)

            # reflection
            xr = x0 + 1*(x0-simplex[idx[2]])
            if func(xr) < f[idx[1]] and func(xr) >= f[idx[0]]:
                simplex[idx[2]] = xr
                continue
            # Expansion
            elif func(xr) < f[idx[0]]:
                xe = x0 + 2*(xr-x0)

                if func(xe) < func(xr):
                    simplex[idx[2]] = xe
                else:
                    simplex[idx[2]] = xr

            # contraction
            elif func(xr) >= f[idx[0]]:
                xc = x0 + 0.5*(simplex[idx[2]] - x0)
                
                if func(xc) < f[idx[2]]:
                    simplex[idx[2]] = xc
                    continue
            
            else:
                for i in idx[1:]:
                    simplex[i] = simplex[idx[0]] + 0.5*(simplex[i] - simplex[idx[0]])
        pt.append(simplex[f.argmin()]) 
        opt.append(f.min())
    pt = np.array(pt)
    opt = np.array(opt)
    return pt[opt.argmin()], opt.min()



if __name__ == "__main__":
    data, min_ = minimize(t.easom,  100)
    print("Minimum value is: %.6f" %min_)
    print("Minimum occurred at: ", data)