import random

import polars as pl
import torch
from tqdm import tqdm

from evaluate import evaluate_output
from run import get_model, run_sample

POPULATION_SIZE = 4
GENERATIONS = 5
LAYERS = 28
PROBLEMS_PER_EVAL = 6
MUTATION_RATE = 0.1

def fitness(num_correct, memory_footprint):
    return num_correct - 0.5 * 1e-9 * (memory_footprint - 8e9)

def format_config(config):
    return "".join(['1' if q else '0' for q in config])

def run_model(config, samples):
    tokenizer, model = get_model(config)
    results = []
    for sample in tqdm(samples):
        question = sample["problem"]
        answer = sample["answer"]
        output = run_sample(tokenizer, model, question)
        student_answer = output.split("assistant\n")[-1].strip()
        results.append((question, answer, student_answer))

    num_correct = 0
    for question, answer, student_answer in results:
        score = evaluate_output(question, answer, student_answer)
        num_correct += score

    memory_footprint = model.get_memory_footprint()

    return num_correct, memory_footprint

def run_population(population, samples):
    results = []
    for config in population:
        num_correct, memory_footprint = run_model(config, samples)
        fit = fitness(num_correct, memory_footprint)
        results.append(fit)
    torch.cuda.empty_cache()
    return results

def main():
    MATH500 = pl.read_ndjson("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl").to_dicts()

    population = []
    for _ in range(POPULATION_SIZE):
        config = [random.choice([True, False]) for _ in range(LAYERS)]
        population.append(config)

    for generation in range(GENERATIONS):
        print(f"Generation {generation}")
        for config in population:
            print(f"  {format_config(config)}")
        samples = random.sample(MATH500, PROBLEMS_PER_EVAL)
        results = run_population(population, samples)
        for config, fit in zip(population, results):
            print(f"  {format_config(config)}: Fitness = {fit}")

        # Select the top half of the population
        sorted_population = [
            config for _, config in sorted(
                zip(results, population), reverse=True
            )
        ]
        population = sorted_population[:POPULATION_SIZE // 2]

        # Cross over
        while len(population) < POPULATION_SIZE:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            crossover_point = random.randint(1, LAYERS - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            population.append(child)

        # Mutate
        for i in range(len(population)):
            for j in range(LAYERS):
                if random.random() < MUTATION_RATE:
                    population[i][j] = not population[i][j]

if __name__ == "__main__":
    main()
