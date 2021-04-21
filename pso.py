import numpy as np

# Todo Implement Particle Swarm Optimization Algorithm
# https://en.wikipedia.org/wiki/Particle_swarm_optimization

def initVelocity(size, dims, bounds):
    l = -abs(bounds[1] - bounds[0])
    h = abs(bounds[1] - bounds[0])
    velocity = np.random.uniform(l, h, (size, dims))
    return velocity

def initParticles(size,dims, bounds):
    particles = np.random.uniform(bounds[0],bounds[1], (size, dims))
    return particles

def globalBest(func, g, particles):
    g_new = np.apply_along_axis(func, 1, particles)
    if func(g) > g_new.min(): 
        return particles[g_new.argmin()]
    else:
        return g

def particleBest(func, p, particles):
    p_new = np.apply_along_axis(func, 1, particles)
    _, d = particles.shape
    p_new = np.tile(p_new,(d,1)).T
    p_old = np.apply_along_axis(func, 1, p)
    p_old = np.tile(p_old,(d,1)).T
    p = np.where(p_new < p_old, particles, p)
    return p

def updateVelocity(velocity, particles, p, g, w, wp, wg):
    s, d = velocity.shape
    rp = np.random.rand(s, d)
    rg = np.random.rand(s, d)
    glob = g.copy()
    v_new = w*velocity + wp*rp*(p-particles) + wg*rg*(glob-particles)

    return v_new

def updateParticle(velocity, particles, lr):
    particle = particles.copy()
    particle += lr*velocity
    return particle

def keepInBounds(bounds, particles):
    l = bounds[0]
    h = bounds[1]
    particles = np.where(particles < l, l, particles)
    particles = np.where(particles > h, h, particles)
    return particles
    
def minimize(func, w, wg, lr, bounds, size, dims, max_iter = 10):
    particles = initParticles(size, dims, bounds)
    p = particles.copy()
    opt = np.apply_along_axis(func, 1, particles)
    g = particles.copy()[opt.argmin()]
    velocity = initVelocity(size,dims, bounds)
    wp = 1-wg
    g_array = [func(g)]
    for i in range(max_iter):
        velocity = updateVelocity(velocity, particles, p, g, w, wp, wg)
        particles = updateParticle(velocity, particles, lr)
        particles = keepInBounds(bounds, particles)
        p = particleBest(func, p, particles)
        g = globalBest(func, g, particles)
        g_array.append(func(g))
    return g, func(g), g_array


if __name__ == "__main__":
    import testFunction as t
    sol, opt, opt_array = minimize(t.goldstein,0.3,0.8,1, [-2,2], 100, 2)
    print(sol, opt)

