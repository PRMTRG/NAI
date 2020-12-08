import math
import random
import time
import matplotlib.pyplot as plt
import sys


GENOTYPE_LENGTH = 128
GENOTYPE_LENGTH_HALF = int( GENOTYPE_LENGTH / 2)


class TestFunction:
    def __init__(self, function, global_optimum, search_domain):
        self.function = function
        self.global_optimum = global_optimum
        self.search_domain = search_domain
    def goal(self, args):
        return abs(self.global_optimum - self.function(args))
    def generate_random_args(self):
        args = []
        for d in self.search_domain:
            args.append(random.uniform(d[0], d[1]))
        return args        


class RandomSolver:
    def __init__(self, test_function, function_name):
        self.test_function = test_function
        self.function_name = function_name
    def solve(self, iterations, save_data_for_plot=False, plotting_step=1):
        best_args = self.test_function.generate_random_args()
        best_result = self.test_function.goal(best_args)
        if save_data_for_plot:
            plot_time = []
            plot_time.append(0)
            plot_result = []
            plot_result.append(best_result)
            start_time = time.perf_counter()
        for i in range(1, iterations):
            new_args = self.test_function.generate_random_args()
            new_result = self.test_function.goal(new_args)
            if new_result < best_result:
                best_args = new_args
                best_result = new_result
            if save_data_for_plot and (i % plotting_step == 0 or i == iterations - 1):
                plot_time.append(time.perf_counter() - start_time)
                plot_result.append(best_result)
        if save_data_for_plot:
            return best_args, best_result, plot_time, plot_result
        else:
            return best_args, best_result
    def solve_multiple_times_and_plot(self, iterations, experiments, plotting_step=1):
        list_plot_time = []
        list_plot_result = []
        for i in range(experiments):
            _, _, plot_time, plot_result = self.solve(iterations, save_data_for_plot=True, plotting_step=plotting_step)
            list_plot_time.append(plot_time)
            list_plot_result.append(plot_result)
        average_out_and_plot(self.function_name, "Random", list_plot_time, list_plot_result)


class RandomClimbSolver:
    def __init__(self, test_function, function_name):
        self.test_function = test_function
        self.function_name = function_name
    def generate_random_neighbour(self, args):
        new_args = []
        for i in range(len(args)):
            domain = self.test_function.search_domain[i]
            max_distance_half = abs(domain[0] - domain[1]) / 100
            new_arg = args[i] + random.uniform(-max_distance_half, max_distance_half)
            if new_arg < domain[0]:
                new_arg = domain[0]
            elif new_arg > domain[1]:
                new_arg = domain[1]
            new_args.append(new_arg)
        return new_args
    def solve(self, iterations, save_data_for_plot=False, plotting_step=1):
        best_args = self.test_function.generate_random_args()
        best_result = self.test_function.goal(best_args)
        if save_data_for_plot:
            plot_time = []
            plot_time.append(0)
            plot_result = []
            plot_result.append(best_result)
            start_time = time.perf_counter()
        for i in range(1, iterations):
            new_args = self.generate_random_neighbour(best_args)
            new_result = self.test_function.goal(new_args)
            if new_result < best_result:
                best_args = new_args
                best_result = new_result
            if save_data_for_plot and (i % plotting_step == 0 or i == iterations - 1):
                plot_time.append(time.perf_counter() - start_time)
                plot_result.append(best_result)
        if save_data_for_plot:
            return best_args, best_result, plot_time, plot_result
        else:
            return best_args, best_result
    def solve_multiple_times_and_plot(self, iterations, experiments, plotting_step=1):
        list_plot_time = []
        list_plot_result = []
        for i in range(experiments):
            _, _, plot_time, plot_result = self.solve(iterations, save_data_for_plot=True, plotting_step=plotting_step)
            list_plot_time.append(plot_time)
            list_plot_result.append(plot_result)
        average_out_and_plot(self.function_name, "Random Climb", list_plot_time, list_plot_result)


class RandomGenotypeSolver:
    def __init__(self, test_function, function_name):
        self.test_function = test_function
        self.function_name = function_name
    def solve(self, iterations, save_data_for_plot=False, plotting_step=1):
        best_genotype = generate_random_genotype()
        best_args = decode_genotype(best_genotype, test_function.search_domain)
        best_fitness = fitness(test_function, best_args)
        if save_data_for_plot:
            plot_time = []
            plot_time.append(0)
            plot_result = []
            plot_result.append(test_function.goal(best_args))
            plot_fitness = []
            plot_fitness.append(best_fitness)
            start_time = time.perf_counter()
        for i in range(1, iterations):
            new_genotype = generate_random_genotype()
            new_args = decode_genotype(new_genotype, test_function.search_domain)
            new_fitness = fitness(test_function, new_args)
            if new_fitness > best_fitness:
                best_genotype = new_genotype
                best_args = new_args
                best_fitness = new_fitness
            if save_data_for_plot and (i % plotting_step == 0 or i == iterations - 1):
                plot_time.append(time.perf_counter() - start_time)
                plot_result.append(test_function.goal(best_args))
                plot_fitness.append(best_fitness)
        if save_data_for_plot:
            return best_genotype, best_args, best_fitness, plot_time, plot_result, plot_fitness
        else:
            return best_genotype, best_args, best_fitness
    def solve_multiple_times_and_plot(self, iterations, experiments, plotting_step=1):
        list_plot_time = []
        list_plot_result = []
        list_plot_fitness = []
        for i in range(experiments):
            _, _, _, plot_time, plot_result, plot_fitness = self.solve(iterations, save_data_for_plot=True, plotting_step=plotting_step)
            list_plot_time.append(plot_time)
            list_plot_result.append(plot_result)
            list_plot_fitness.append(plot_fitness)
        average_out_and_plot(self.function_name, "Random Genotype", list_plot_time, list_plot_result) 
        average_out_and_plot(self.function_name, "Random Genotype", list_plot_time, list_plot_fitness)  


def average_out_and_plot(function_name, solver_name, list_plot_time, list_plot_result):
    n = len(list_plot_time[0])
    m = len(list_plot_time)
    plot_time = []
    plot_result = []
    for i in range(n):
        s1 = 0
        s2 = 0
        for j in range(m):
            s1 += list_plot_result[j][i]
            s2 += list_plot_time[j][i]
        plot_result.append(s1 / m)
        plot_time.append(s2 / m)
    plt.plot(plot_time, plot_result)
    plt.title("{} - {}".format(function_name, solver_name))
    plt.xlabel('Time')
    plt.ylabel('Result')
    plt.show()


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


def generate_sphere_function(n):
    def sphere_formula(x):
        n = len(x)
        s = 0
        for i in range(n):
            s += x[i] ** 2
        return s
    global_optimum = 0
    search_domain = []
    for i in range(n):
        search_domain.append((-100, 100))
    return sphere_formula, global_optimum, search_domain


def generate_beale_function():
    def beale_formula(x):
        return (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    global_optimum = 0
    search_domain = [ (-4.5, 4.5), (-4.5, 4.5) ]
    return beale_formula, global_optimum, search_domain


def generate_booth_function():
    def booth_formula(x):
        return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    global_optimum = 0
    search_domain = [ (-10, 10), (-10, 10) ]
    return booth_formula, global_optimum, search_domain


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
    max_num = 2 ** GENOTYPE_LENGTH_HALF
    tot_x = search_domain[0][1] - search_domain[0][0]
    tot_y = search_domain[1][1] - search_domain[1][0]
    x_gen = genotype[:GENOTYPE_LENGTH_HALF]
    y_gen = genotype[GENOTYPE_LENGTH_HALF:]
    x_num = gray_to_decimal(x_gen)
    y_num = gray_to_decimal(y_gen)
    x = search_domain[0][0] + ( ( tot_x / max_num ) * x_num )
    y = search_domain[1][0] + ( ( tot_y / max_num ) * y_num )
    return [ x, y ]


def encode_genotype(args, search_domain):
    max_num = 2 ** GENOTYPE_LENGTH_HALF
    tot_x = search_domain[0][1] - search_domain[0][0]
    tot_y = search_domain[1][1] - search_domain[1][0]
    x = args[0]
    y = args[1]
    x_num = round(( x - search_domain[0][0] ) / ( tot_x / max_num ))
    y_num = round(( y - search_domain[1][0] ) / ( tot_y / max_num ))
    x_gen = decimal_to_gray(x_num)
    y_gen = decimal_to_gray(y_num)
    if len(x_gen) > GENOTYPE_LENGTH_HALF:
        x_gen = x_gen[:GENOTYPE_LENGTH_HALF]
    if len(y_gen) > GENOTYPE_LENGTH_HALF:
        y_gen = y_gen[:GENOTYPE_LENGTH_HALF]
    for i in range(len(x_gen), GENOTYPE_LENGTH_HALF):
        x_gen.append(0)
    for i in range(len(y_gen), GENOTYPE_LENGTH_HALF):
        y_gen.append(0)
    return x_gen + y_gen


def generate_random_genotype():
    x_gen = []
    y_gen = []
    for i in range(GENOTYPE_LENGTH_HALF):
        x_gen.append(random.getrandbits(1))
        y_gen.append(random.getrandbits(1))
    return x_gen + y_gen


def fitness(test_function, args):
    return 1 / test_function.goal(args)


if __name__ == "__main__":

    # search_domain = [ (-10, 10), (-10, 10) ]
    # genotype = generate_random_genotype()
    # genotype = [ 0 for i in range(GENOTYPE_LENGTH)]
    # genotype[GENOTYPE_LENGTH_HALF - 2] = 0
    # genotype[GENOTYPE_LENGTH_HALF - 1] = 0
    # genotype[GENOTYPE_LENGTH - 2] = 0
    # genotype[GENOTYPE_LENGTH - 1] = 1
    # args = decode_genotype(genotype, search_domain)
    # reencoded_genotype = encode_genotype(args, search_domain)
    # args_after_reencoding = decode_genotype(reencoded_genotype, search_domain)
    
    # print("Genotype:")
    # print(genotype[:GENOTYPE_LENGTH_HALF])
    # print(genotype[GENOTYPE_LENGTH_HALF:])
    # print("Reencoded genotype:")
    # print(reencoded_genotype[:GENOTYPE_LENGTH_HALF])
    # print(reencoded_genotype[GENOTYPE_LENGTH_HALF:])
    # print("Args:")
    # print(args)
    # print("Args after reencoding:")
    # print(args_after_reencoding)
    
    
    functions = []
    functions.append([ generate_rastrigin_function(2), "Rastrigin function" ])
    functions.append([ generate_sphere_function(2), "Sphere function" ])
    functions.append([ generate_beale_function(), "Beale function" ])
    functions.append([ generate_booth_function(), "Booth function" ])
    
    random_solvers = []
    for f in functions:
        test_function = TestFunction(f[0][0], f[0][1], f[0][2])
        random_solver = RandomSolver(test_function, f[1])
        random_solvers.append(random_solver)
        
    random_climb_solvers = []
    for f in functions:
        test_function = TestFunction(f[0][0], f[0][1], f[0][2])
        random_climb_solver = RandomClimbSolver(test_function, f[1])
        random_climb_solvers.append(random_climb_solver)
    
    random_genotype_solvers = []
    random_genotype_solvers.append(RandomGenotypeSolver(generate_rastrigin_function(2), "Rastrigin function"))
    
    for rs in random_solvers:
        rs.solve_multiple_times_and_plot(100, 20, plotting_step=1)
    
    for rcs in random_climb_solvers:
        rcs.solve_multiple_times_and_plot(100, 20, plotting_step=1)

    for rgs in random_genotype_solvers:
        rgs.solve_multiple_times_and_plot(100, 20, plotting_step=1)
    