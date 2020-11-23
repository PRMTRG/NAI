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
    def __init__(self, test_function):
        self.test_function = test_function
    def solve(self):
        args = []
        for arg in self.test_function.search_domain:
            args.append(random.uniform(arg[0], arg[1]))
        return args, self.test_function.goal(args)
    def solve_n_times(self, n):
        best_solution, best_result = self.solve()
        for i in range(1, n):
            new_solution, new_result = self.solve()
            if new_result < best_result:
                best_solution = new_solution.copy()
                best_result = new_result
        return best_solution, best_result
    def solve_n_times_and_save_data_for_plot(self, n, step):
        plot_best_results = []
        plot_time_taken = []
        best_solution, best_result = self.solve()
        plot_best_results.append(best_result)
        plot_time_taken.append(0)
        n -= 1
        start_time = time.perf_counter()
        while n > 0:
            if n - step < 0:
                step = n
            n -= step
            new_best_solution, new_best_result = self.solve_n_times(step)
            if new_best_result < best_result:
                best_solution = new_best_solution
                best_result = new_best_result
            plot_best_results.append(best_result)
            plot_time_taken.append(time.perf_counter() - start_time)
        return best_solution, best_result, plot_best_results, plot_time_taken
    def solve_n_times_m_times_and_plot(self, n, m, step, plot_title):
        best_results_list = []
        time_taken_list = []
        plot_results = []
        plot_time = []
        for i in range(m):
            _, _, best_results, time_taken = self.solve_n_times_and_save_data_for_plot(n, step)
            best_results_list.append(best_results)
            time_taken_list.append(time_taken)
        n = math.ceil(n / step)
        for i in range(n):
            s1 = 0
            s2 = 0
            for j in range(m):
                s1 += best_results_list[j][i]
                s2 += time_taken_list[j][i]
            plot_results.append(s1 / m)
            plot_time.append(s2 / m)
        plt.plot(plot_time, plot_results)
        plt.title(plot_title)
        plt.xlabel('Time')
        plt.ylabel('Result')
        plt.show()
        

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
    rastring_formula, rastring_global_optimum, rastring_search_domain = generate_rastring_function(10)
    rastring_random_solver = RandomSolver(TestFunction(rastring_formula, rastring_global_optimum, rastring_search_domain))
    
    sphere_formula, sphere_global_optimum, sphere_search_domain = generate_sphere_function(10)
    sphere_random_solver = RandomSolver(TestFunction(sphere_formula, sphere_global_optimum, sphere_search_domain))
    
    beale_formula, beale_global_optimum, beale_search_domain = generate_beale_function()
    beale_random_solver = RandomSolver(TestFunction(beale_formula, beale_global_optimum, beale_search_domain))
    
    booth_formula, booth_global_optimum, booth_search_domain = generate_booth_function()
    booth_random_solver = RandomSolver(TestFunction(booth_formula, booth_global_optimum, booth_search_domain))
    
    rastring_random_solver.solve_n_times_m_times_and_plot(100, 20, 1, "Rastring function")
    sphere_random_solver.solve_n_times_m_times_and_plot(100, 20, 1, "Sphere function")
    beale_random_solver.solve_n_times_m_times_and_plot(100, 20, 1, "Beale function")
    booth_random_solver.solve_n_times_m_times_and_plot(100, 20, 1, "Booth function")
    
    