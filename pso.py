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

    v_new = w*velocity + wp*rp*(p-particles) + wg*rg*(g-particles)

    return v_new

def updateParticle(velocity, particles, lr):
    particles += lr*velocity
    return particles

def keepInBounds(bounds, particles):
    l = bounds[0]
    h = bounds[1]
    particles = np.where(particles < l, l, particles)
    particles = np.where(particles > h, h, particles)
    return particles
    
def minimize(func, w, wp, wg, lr, bounds, size, dims, max_iter = 10):
    particles = initParticles(size, dims, bounds)
    p = particles.copy()
    opt = np.apply_along_axis(func, 1, particles)
    g = particles[opt.argmin()]
    velocity = initVelocity(size,dims, bounds)

    for i in range(max_iter):
        velocity = updateVelocity(velocity, particles, p, g, w, wp, wg)
        particles = updateParticle(velocity, particles, lr)
        particles = keepInBounds(bounds, particles)
        p = particleBest(func, p, particles)
        g = globalBest(func, g, particles)

    return g


if __name__ == "__main__":
    import testFunction as t
    sol = minimize(t.beales,0.3,0.5,0.8,1, [-4.5,4.5], 100, 2)
    opt = t.beales(sol)
    print(opt, sol)

