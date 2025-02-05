import matplotlib.pyplot as plt
import random
from collections import defaultdict, namedtuple

Operation = namedtuple('Operation', ['job_id', 'machine_id', 'duration'])


class Job:
    def __init__(self, job_id, operations):
        self.job_id = job_id
        self.operations = operations


def plot_schedule(schedules):
    fig, ax = plt.subplots(figsize=(10, 5))
    machines = sorted(schedules.keys())
    colors = plt.cm.viridis(range(0, 256, 256 // len(machines)))

    for i, (machine, operations) in enumerate(schedules.items()):
        for start, end, job_id in operations:
            ax.barh(y=machine, width=end - start, left=start, height=0.4, color=colors[i], edgecolor='black')
            ax.text(x=start + (end - start) / 2, y=machine, s=f'J{job_id}', color='white', va='center', ha='center')

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks([m for m in machines])
    ax.set_yticklabels([f'Machine {m}' for m in machines])
    ax.set_title('Job Scheduling Gantt Chart')
    plt.tight_layout()
    plt.show()


def initialize_population(size, jobs):
    population = []
    for _ in range(size):
        individual = []
        for job in jobs:
            individual.append(job.operations)  # Ensuring job operation order is maintained
        random.shuffle(individual)
        population.append(individual)
    return population


def calculate_fitness(chromosome):
    machine_end_times = defaultdict(int)
    job_next_start_times = defaultdict(int)
    job_completion_times = defaultdict(int)
    schedules = defaultdict(list)

    for job_operations in chromosome:
        for op in job_operations:
            start_time = max(machine_end_times[op.machine_id], job_next_start_times[op.job_id])
            end_time = start_time + op.duration
            machine_end_times[op.machine_id] = end_time
            job_next_start_times[op.job_id] = end_time
            schedules[op.machine_id].append((start_time, end_time, op.job_id))
            job_completion_times[op.job_id] = end_time

    fitness = max(machine_end_times.values())
    return fitness, schedules, job_completion_times


def crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent1[cut:]
    return child1, child2


def mutate(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]


def genetic_algorithm(jobs, population_size, generations, mutation_rate=0.1):
    population = initialize_population(population_size, jobs)
    best_fitness = float('inf')
    best_chromosome = None
    best_schedules = None
    best_job_completion_times = None

    for generation in range(generations):
        population_fitness = []
        for chromosome in population:
            fitness, schedules, job_completion_times = calculate_fitness(chromosome)
            population_fitness.append((chromosome, fitness))

            if fitness < best_fitness:
                best_fitness = fitness
                best_chromosome = chromosome
                best_schedules = schedules
                best_job_completion_times = job_completion_times

        population_fitness.sort(key=lambda x: x[1])
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population_fitness[:population_size // 2], k=2)
            child1, child2 = crossover(parent1[0], parent2[0])

            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:population_size]
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    return best_chromosome, best_schedules, best_fitness, best_job_completion_times


def read_jobs_from_file(filename):
    jobs = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(';')
            job_id = int(parts[0])
            operations = []
            for op in parts[1:]:
                machine_id, duration = map(int, op.split(','))
                operations.append(Operation(job_id, machine_id, duration))
            jobs.append(Job(job_id, operations))
    return jobs


def get_jobs_from_user():
    jobs = []
    while True:
        job_id = int(input("Enter Job ID: "))
        operations = []
        while True:
            machine_id = int(input("Enter Machine ID for an operation (or -1 to finish this job): "))
            if machine_id == -1:
                break
            duration = int(input("Enter Duration for this operation: "))
            operations.append(Operation(job_id, machine_id, duration))
        jobs.append(Job(job_id, operations))
        cont = input("Do you want to add another job? (y/n): ")
        if cont.lower() != 'y':
            break
    return jobs


def main():
    print("Choose input method:")
    print("1. Read jobs from a file")
    print("2. Enter jobs manually")
    choice = int(input("Enter choice (1 or 2): "))

    if choice == 1:

        jobs = read_jobs_from_file('input.txt')
    elif choice == 2:
        jobs = get_jobs_from_user()
    else:
        print("Invalid choice")
        return

    best_overall_fitness = float('inf')
    best_overall_schedules = None
    best_overall_job_completion_times = None

    loop_range = 20

    for run in range(loop_range):
        best_solution, schedules, best_fitness, job_completion_times = genetic_algorithm(jobs, population_size=100, generations=50)
        print(f"\nRun {run + 1}, Best Fitness: {best_fitness}")

        print(f"Schedule for Run {run + 1}:")
        for machine, ops in schedules.items():
            job_order = [f"Job {job_id} from {start} to {end}" for start, end, job_id in ops]
            print(f"Machine {machine}: " + " -> ".join(job_order))

        if best_fitness < best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_schedules = schedules
            best_overall_job_completion_times = job_completion_times

    print("_" * 100)
    print("Best Fitness Schedule:")
    for machine, ops in best_overall_schedules.items():
        for start, end, job_id in ops:
            print(f"Machine {machine}: Job {job_id} from {start} to {end}")

    # Print the order of job completion
    print("\nOrder of job completion based on the best schedule:")
    sorted_jobs = sorted(best_overall_job_completion_times.items(), key=lambda x: x[1])
    for job_id, completion_time in sorted_jobs:
        print(f"Job {job_id} completed at time {completion_time}")

    print(f"\n--> Best Overall Fitness: {best_overall_fitness} check the path in the Gantt Chart")
    plot_schedule(best_overall_schedules)


if __name__ == "__main__":
    main()
