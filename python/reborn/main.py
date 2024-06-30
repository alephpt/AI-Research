if __name__ == "__main__":
    from grid import Grid
    from genetic import Genetic

    env = Grid()
    ga = Genetic(
        pop_size=100,
        gene_length=10,
        crossover_rate=0.7,
        mutation_rate=0.01,
        generations=1000,
        env=env,
    )

    ga.train()