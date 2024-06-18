import numpy as np
import random

GENDERS = ["Male", "Female"]
REPRODUCTIVE_CHANGE = {
    "Male": 0.4,
    "Female": 0.8
}

class Individual:
    def __init__(self, id, wealth=0, satisfaction=0, generation=0):
        self.id = id
        self.age = 0
        self.fitness = 0
        self.energy = 100
        self.generation = generation
        self.gender = np.random.choice(GENDERS)
        self.satisfaction = satisfaction
        self.wealth = wealth
        self.sex_appeal = self.factorSexAppeal()
        self.offspring = []
        self.father = None
        self.mother = None
        self.alive = True

    def persist(self):
        self.alive = False if self.energy < 0 else True
        
    def factorSexAppeal(self):
        return (self.fitness * 0.5 + self.wealth * 0.5 + self.satisfaction * 0.5 + self.energy * 0.5) // 2
        
    def work(self):
        print("[WORK]: ", self.id, " is working")
        self.age += 1
        self.wealth += 10
        self.energy -= 10
        self.fitness += 5
        self.satisfaction += 5
        
    def eat(self):
        print("[EAT]: ", self.id, " is eating")
        self.age += 1
        self.wealth -= 10
        self.energy += 10
        self.fitness -= 5
        self.satisfaction += 5
        
    def sleep(self):
        print("[SLEEP]: ", self.id, " is sleeping")
        self.age += 3
        self.energy += 30
        self.satisfaction += np.random.randint(-10, 10)
        return
    
    def attractedTo(self, partner):
        return True if ((self.sex_appeal < (partner.sex_appeal * 1.5)) and (self.sex_appeal > (partner.sex_appeal / 1.5))) else False
    
    def chanceToReproduce(self, partner):
        return True if np.random.uniform(0, 1) < 0.5 else False
    
    def conception(self, partner):
        if not self.chanceToReproduce(partner):
            print("[SEX]: ", self.id, "has no chance to reproduce with partner", partner.id)
            return
        
        offspring = Individual(
            id=None,
            wealth=(self.wealth + partner.wealth) // 10,
            satisfaction=(self.satisfaction + partner.satisfaction) // 10,
            generation=max(self.generation, partner.generation) + 1
        )
        
        self.age += 1
        self.satisfaction += np.random.randint(-10, 10)
        self.energy -= np.random.randint(10, 20)
        self.wealth -= self.wealth // 10
        
        partner.age += 1
        partner.satisfaction += np.random.randint(-10, 10)
        partner.energy -= np.random.randint(10, 20)
        partner.wealth -= partner.wealth // 10
        
        print("[SEX]: ", self.id, "concieved with partner", partner.id, "and had a baby ", offspring.gender)
        
        return offspring
    
    def haveSex(self, partner):
        self.age += 1
        partner.age += 1
        
        combined_energy = self.energy + partner.energy
        explosive_energy = self.energy + partner.energy ** 2
        self.energy += random.uniform(-explosive_energy, combined_energy)
        partner.energy += random.uniform(-explosive_energy, combined_energy)
        
        if not self.attractedTo(partner):
            print("[SEX]: ", self.id, "and", partner.id, "are not attracted to each other")
            self.satisfaction -= random.uniform(-20, 5)
            partner.satisfaction -= random.uniform(-20, 5)
            return
        
        self.satisfaction += np.random.randint(-10, 20)
        partner.satisfaction += np.random.randint(-10, 20)
        
        if self.gender == partner.gender:
            print("[SEX]: ", self.id, "and", partner.id, "are the same sex and cannot reproduce")
            return

        return self.conception(partner)        

class Population:
    def __init__(self, size=10):
        self.size = size
        self.individuals = [Individual(id=i) for i in range(size)]
    
    def isExtinct(self):
        return all([not individual.alive for individual in self.individuals])
    
    def selectTopIndividuals(self, top):
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:top]
    
    def simulate(self):
        for individual in self.individuals:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Individual: ", individual.id)
            print("Age: ", individual.age)
            print("Energy: ", individual.energy)
            print("Wealth: ", individual.wealth)
            print("Satisfaction: ", individual.satisfaction)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        
            if not individual.alive:
                continue
            
            action = np.random.choice([individual.work, individual.eat, individual.sleep, individual.haveSex]) ## Need to improve this to use Q-Learning
            
            if action == individual.haveSex:
                partner = np.random.choice(self.individuals)
                
                if partner == individual:
                    continue
                
                offspring = action(partner)
                
                if offspring:
                    offspring.id = len(self.individuals)
                    offspring.father = individual if individual.gender == "Male" else partner
                    offspring.mother = individual if individual.gender == "Female" else partner                    
                    
                    individual.offspring.append(offspring)
                    partner.offspring.append(offspring)
                    
                    self.individuals.append(offspring)
            else:
                action()
            
            individual.persist()
                
    def evolve(self):
        top_individuals = self.selectTopIndividuals(10)
        
        self.individuals = [Individual(
            id=individual.id,
            wealth=individual.wealth,
            satisfaction=individual.satisfaction,
            generation=individual.generation
        ) for individual in top_individuals]
        
        self.size = len(self.individuals)

if __name__ == "__main__":        
    population = Population()

    while not population.isExtinct():
        population.simulate()
        population.evolve()
        
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Population is extinct")
    print("Total individuals: ", len(population.individuals))
    print("Total generations: ", max([individual.generation for individual in population.individuals]))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Max wealth: ", max([individual.wealth for individual in population.individuals]))
    print("Max satisfaction: ", max([individual.satisfaction for individual in population.individuals]))
    print("Max fitness: ", max([individual.fitness for individual in population.individuals]))
    print("Max energy: ", max([individual.energy for individual in population.individuals]))
    print("Max age: ", max([individual.age for individual in population.individuals]))
    print("Max offspring: ", max([len(individual.offspring) for individual in population.individuals]))
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Avg wealth: ", np.mean([individual.wealth for individual in population.individuals]))
    print("Avg satisfaction: ", np.mean([individual.satisfaction for individual in population.individuals]))
    print("Avg fitness: ", np.mean([individual.fitness for individual in population.individuals]))
    print("Avg energy: ", np.mean([individual.energy for individual in population.individuals]))
    print("Avg age: ", np.mean([individual.age for individual in population.individuals]))
    print("Avg offspring: ", np.mean([len(individual.offspring) for individual in population.individuals]))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    