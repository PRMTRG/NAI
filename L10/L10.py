import random
import math
import matplotlib.pyplot as plt


class GA:
    def __init__(self, gen_starting_pop_f, fitness_f, selection_f, crossover_f,
                 mutation_f, termination_f, prob_crossover, prob_mutation, elite_count):
        self.gen_starting_pop_f = gen_starting_pop_f
        self.fitness_f = fitness_f
        self.selection_f = selection_f
        self.crossover_f = crossover_f
        self.mutation_f = mutation_f
        self.termination_f = termination_f
        self.prob_crossover = prob_crossover
        self.prob_mutation = prob_mutation
        self.elite_count = elite_count
        self.population = self.gen_starting_pop_f()
    def get_best_specimen_index(self):
        return self.population_fitness.index(max(self.population_fitness))
    def run(self):
        pop_size = len(self.population)
        self.population_fitness = []
        for i in range(pop_size):
            self.population_fitness.append(self.fitness_f(self.population[i]))
        plot_data = []
        while not self.termination_f(self):         
            # elites
            parents = []
            best_specimen_index = self.get_best_specimen_index()
            for i in range(self.elite_count):
                parents.append(self.population[best_specimen_index])
            
            # selection
            for i in range(len(parents), pop_size):
                parents.append(self.selection_f(self.population, self.population_fitness))
            
            # crossing
            parents_idx = 1
            children = []            
            while len(children) < pop_size and parents_idx < pop_size:
                for child in self.crossover_f(self.prob_crossover, parents[parents_idx],
                                              parents[parents_idx - 1]):
                    children.append(child)
                parents_idx += 2
            if len(children) > pop_size:
                children = children[:pop_size]
            
            # mutation
            for i in range(pop_size):
                children[i] = self.mutation_f(children[i], self.prob_mutation)
            
            self.population = children
            for i in range(pop_size):
                self.population_fitness[i] = self.fitness_f(self.population[i])
            plot_data.append(self.population_fitness[self.get_best_specimen_index()])
        
        for i in range(pop_size):
            print("{} = {}".format(self.population_fitness[i], self.population[i]))
        return self.population[self.get_best_specimen_index()].copy(), plot_data


def gen_gen_starting_pop(population_size, genotype_size):
    def gen_starting_pop():
        population = []
        for i in range(population_size):
            population.append([ random.getrandbits(1) for j in range(genotype_size) ])
        return population
    return gen_starting_pop


def tournament_selection(pop, pop_fit):
    tournament_size = 2
    tournament = []
    for i in range(tournament_size):
        tournament.append(random.randint(0, len(pop) - 1))
    best = tournament[0]
    for i in range(1, tournament_size):
        if pop_fit[tournament[i]] > pop_fit[best]:
            best = tournament[i]
    return pop[best].copy()


def one_point_crossover(prob_crossover, parent_a, parent_b):
    if random.random() > prob_crossover:
        return parent_a, parent_b
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    pp = random.randint(0, len(parent_a))
    for i in range(pp, len(parent_a)):
        child_a[i], child_b[i] = child_b[i], child_a[i]
    return child_a, child_b


def mutation(genotype, prob):
    new_genotype = genotype.copy()
    for i in range(len(new_genotype)):
        if random.random() < prob:
            new_genotype[i] = 1 - new_genotype[i]
    return new_genotype


def gen_terminate_after_iterations(iterations):
    def terminate_after_iterations(self):
        try:
            self.iteration += 1
        except:
            self.iteration = 1
            return False
        if self.iteration >= iterations:
            return True
        return False
    return terminate_after_iterations


def gen_terminate_after_fitness_goal(goal):
    def terminate_after_fitness_goal(self):
        try:
            if sum(self.population_fitness) / len(self.population_fitness) >= goal:
                return True
        except:
            return False
        return False
    return terminate_after_fitness_goal


def gen_terminate_immediately():
    def terminate_immediately(self):
        return True
    return terminate_immediately


def xor_list(a, b):
    if len(a) != len(b):
        raise Exception("xor_list")
    result = []
    for i in range(len(a)):
        result.append(a[i] ^ b[i])
    return result


def shr_list(a, b):
    result = []
    for i in range(len(a) - b):
        result.append(a[i + b])
    for i in range(b):
        result.append(0)
    return result


def gray_to_binary_list(gray):
    result = gray.copy()
    mask = gray.copy()
    while 1 in mask:
        mask = shr_list(mask, 1)
        result = xor_list(result, mask)
    return result


def binary_list_to_gray(binary_list):
    result = xor_list(binary_list, shr_list(binary_list, 1))  
    return result


def binary_list_to_decimal(binary_list):
    result = 0
    for i in range(len(binary_list)):
        result += binary_list[i] * ( 2 ** i )
    return result


def decimal_to_binary_list(number):
    result = [int(i) for i in bin(number)[2:]]
    result.reverse()
    return result


def gray_to_decimal(gray):
    return binary_list_to_decimal(gray_to_binary_list(gray))
    
    
def decimal_to_gray(number):
    return binary_list_to_gray(decimal_to_binary_list(number))


def decode_genotype(genotype, search_domain):
    max_num = 2 ** (len(genotype) // 2)
    tot_x = search_domain[0][1] - search_domain[0][0]
    tot_y = search_domain[1][1] - search_domain[1][0]
    x_gen = genotype[:(len(genotype) // 2)]
    y_gen = genotype[(len(genotype) // 2):]
    x_num = gray_to_decimal(x_gen)
    y_num = gray_to_decimal(y_gen)
    x = search_domain[0][0] + ( ( tot_x / max_num ) * x_num )
    y = search_domain[1][0] + ( ( tot_y / max_num ) * y_num )
    return [ x, y ]


def generate_rastrigin_function(n):
    def rastrigin_formula(x):
        n = len(x)
        a = 10
        s = 0
        for i in range(n):
            s += x[i] ** 2 - a * math.cos(2 * math.pi * x[i])
        return a * n + s
    global_optimum = 0
    search_domain = []
    for i in range(n):
        search_domain.append((-5.12, 5.12))
    return rastrigin_formula, global_optimum, search_domain


def rastrigin_fitness(genotype):
    formula, global_optimum, search_domain = generate_rastrigin_function(2)
    args = decode_genotype(genotype, search_domain)
    return 1 / ( abs(global_optimum - formula(args)) + 1)


if __name__ == "__main__":
    
    iterations = 200
    population_size = 100
    elite_count = 1
    genotype_size = 20
    crossover_probability = 0.5
    mutation_probability = 0.01
    
    ga = GA(gen_gen_starting_pop(population_size, genotype_size),
            rastrigin_fitness, tournament_selection, one_point_crossover,
            mutation, gen_terminate_after_iterations(iterations),
            crossover_probability, mutation_probability, elite_count)
    
    best_genotype, plot_data = ga.run()
    
    formula, _, search_domain = generate_rastrigin_function(2)
    
    print()
    print("{} = {} = {} = {}".format(rastrigin_fitness(best_genotype),
                                     formula(decode_genotype(
                                         best_genotype, search_domain)),
                                     decode_genotype(best_genotype, search_domain),
                                     best_genotype))
    
    plt.plot(plot_data)
    plt.show()
    
    
    