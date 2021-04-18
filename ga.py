import numpy as np
import testFunction as t
# np.random.seed(1)

def initialPopulation(popSize, dims, bounds):
    return np.random.uniform(low = bounds[0], high  = bounds[1], size = (popSize, dims))

def fitness(func, pop):
    return np.apply_along_axis(func, 1, pop)

def selectParents(pop,fit, cnt):
    fit = np.argsort(fit)
    return pop[fit[:cnt]]

def crossOver(parents, size):
    offspring = np.empty((parents.shape[0]*size, parents.shape[1]))
    crossover_point = np.uint8(parents.shape[1]/2)
    for k in range(len(parents)):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutate(offspring, rate):
    for i in range(offspring.shape[0]):
        p = np.random.rand()
        if p <= rate:
            k = np.random.randint(low = 0, high = offspring.shape[1])
            # print(k)
            offspring[i,k] += np.random.uniform(low = -1, high = 1)
    return offspring

def minimize(func, bounds, population_size, offspring_size, generations):
    ipop = initialPopulation(population_size, 2, bounds)
    for i in range(generations):
        
        fit = fitness(func, ipop)
        parents = selectParents(ipop,fit, population_size//2)
        offspring = crossOver(parents, offspring_size)
        offspring = mutate(offspring, 0.5)
        ipop = np.concatenate((offspring, parents))
    fit = fitness(func, ipop)
    return fit.min(), ipop[fit.argmin()]

def ga(simulations, func, bounds, population_size, offspring_size, generations):
    opt, pt = [], []
    for i in range(simulations):
        optimal, point = minimize(func, bounds, population_size, offspring_size, generations)
        opt.append(optimal)
        pt.append(point)
    opt = np.array(opt)
    pt = np.array(pt)
    return opt, pt

if __name__ == "__main__":
    opt, pt = ga(30, t.beales, [-4.5,4.5], 30, 1, 10)
    print(opt.min(), pt[opt.argmin()])