import math
import random
import time
import matplotlib.pyplot as plt


class TestFunction:
    def __init__(self, function, global_optimum, search_domain):
        self.function = function
        self.global_optimum = global_optimum
        self.search_domain = search_domain
    def goal(self, args):
        return abs(self.global_optimum - self.function(args))


class RandomSolver:
    def __init__(self, test_function, name):
        self.test_function = test_function
        self.name = name
    def generate_random_args(self):
        args = []
        for d in self.test_function.search_domain:
            args.append(random.uniform(d[0], d[1]))
        return args
    def solve(self, iterations, save_data_for_plot=False, plotting_step=1):
        best_args = self.generate_random_args()
        best_result = self.test_function.goal(best_args)
        if save_data_for_plot:
            plot_time = []
            plot_time.append(0)
            plot_result = []
            plot_result.append(best_result)
            start_time = time.perf_counter()
        for i in range(1, iterations):
            new_args = self.generate_random_args()
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
    def average_out_and_plot(self, list_plot_time, list_plot_result):
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
        plt.title(self.name + " - Random")
        plt.xlabel('Time')
        plt.ylabel('Result')
        plt.show()
    def solve_multiple_times_and_plot(self, iterations, experiments, plotting_step=1):
        list_plot_time = []
        list_plot_result = []
        for i in range(experiments):
            _, _, plot_time, plot_result = self.solve(iterations, save_data_for_plot=True, plotting_step=plotting_step)
            list_plot_time.append(plot_time)
            list_plot_result.append(plot_result)
        self.average_out_and_plot(list_plot_time, list_plot_result)


class RandomClimbSolver:
    def __init__(self, test_function, name):
        self.test_function = test_function
        self.name = name
    def generate_random_args(self):
        args = []
        for d in self.test_function.search_domain:
            args.append(random.uniform(d[0], d[1]))
        return args
    def generate_random_neighbour(self, args):
        new_args = []
        for i in range(len(args)):
            domain = self.test_function.search_domain[i]
            max_distance_half = abs(domain[0] - domain[1]) / 100
            new_args.append(args[i] + random.uniform(-max_distance_half, max_distance_half))
        return new_args
    def solve(self, iterations, save_data_for_plot=False, plotting_step=1):
        best_args = self.generate_random_args()
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
    def average_out_and_plot(self, list_plot_time, list_plot_result):
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
        plt.title(self.name + " - Random Climb")
        plt.xlabel('Time')
        plt.ylabel('Result')
        plt.show()
    def solve_multiple_times_and_plot(self, iterations, experiments, plotting_step=1):
        list_plot_time = []
        list_plot_result = []
        for i in range(experiments):
            _, _, plot_time, plot_result = self.solve(iterations, save_data_for_plot=True, plotting_step=plotting_step)
            list_plot_time.append(plot_time)
            list_plot_result.append(plot_result)
        self.average_out_and_plot(list_plot_time, list_plot_result)


def generate_rastring_function(n):
    def rastring_formula(x):
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
    return rastring_formula, global_optimum, search_domain


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


if __name__ == "__main__":
    
    functions = []
    functions.append([ generate_rastring_function(2), "Rastring function" ])
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
    
    for rs in random_solvers:
        rs.solve_multiple_times_and_plot(100, 20, plotting_step=1)
    
    for rcs in random_climb_solvers:
        rcs.solve_multiple_times_and_plot(100, 20, plotting_step=1)

    
    