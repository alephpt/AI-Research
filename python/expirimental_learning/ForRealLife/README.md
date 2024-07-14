# Creating The Matrix
## Deep Genetic Learning Outline

## Goal
 - Determine if an Unit can figure out how to survive, if the only two options are Eating and working - and the own two indicators of state are intiailly energy and money.
 - Once an unit is able to survive, they should learn how to reproduce.
 - Reproduction introduces the concept of a relationship, influence or attraction
 - At this point we can scale work/eating into 'trade' with a deals 'cost':'return' being determined by their level of influence.
 - At this point we can scale the unit to be able to create their own markets and homes base on their money + food and money + energy, respectively.
 - We want to start with one, and allow for the unit to explore options of planting food genetically to see if they make a correlation as a society, or on accident.

### Concept:
  - Can we simulate society without telling the unit how to exist within it?
  - Can we define rules for a society 'member' to exist within?
  - What sort of free units would emerge in society e.g. 
        what would be varrying 'optimal' ways of "scoring high"?

Visit the [Outline](https://github.com/alephpt/AI-Research/blob/main/python/expirimental_learning/ForRealLife/outline.md) for the full writeup. (subject to change)

Brief:
### Rules:
- Units require food to Survive
- Food costs 5 Money and provide 10 Energy
- Work costs 5 Energy and provide 10 Money
- Units can reproduce with other Units to create new Units if M+F
- Units can die if they run out of Energy
- Units can be male or female
- Any Unit can be Mate but only Male and Female Units can Reproduce, at an appropriated chance
- Mating brings happiness
- Reproducing brings a large happiness reward
- The 'Reinforcement' Q Network allows for units to cross-breed, and potentially share information within a 'family'
- Epochs end when the entire population dies.
- Each Unit will have their own Table with trial and error, and the child will inherit a mix of their parents genes.
- The top happiness, wealth, energy and age, as well as the longest living generation are used to repopulate, with a shuffled mix of the worst performers

#### Deep Reinforcement Learning for Units Behavior
   - Learn to Navigate
   - Learn to Survive
   - Learn to Reproduce
   - Reward Units based on 'Happiness', second to 'Survival'
       - Reward = Lifespan + Happiness
   - Can we multi-sample async training buffers for 'higher resolution' ?

#### Generations of Units
- Genetic Algorithm for Population Evolution
   -  Fill half of the population, with half of the furthest generation


Working Space:
## TODOs
 -[X] Write paper on the concept of genetic reinforcement
 -[X] Define Policies
 -[X] Add an 'Impulse Factor' to the Unit
 -[X] Be able to 'hone' target
 -[ ] Be able to acheive interaction with a cell
 -[ ] Implement 'Random' choices
 -[ ] Implement 'Sleep' with No Reward except Sleep and Age (and 'refreshing' Fatigue (tbd))
 -[ ] Remove Home and implement Building
 -[ ] Configure Table for State: Action determination
    -[ ] Integrate State as Key
    -[ ] Define Action Table
    -[ ] Determine Reward factor for for 'Best Action'
 -[ ] Make type safe
 -[ ] Write Nova Bindings

### Future Potential Implementations:
- Jobs provide Income for Some and Food for Others.. 
    by implementing a chance to trade, 
    and allowing them to pick one or the other : Vendor or Buyer
   - Vendor makes Money, and Buyer gets Food
        - Later, Banks can let you Save
        - Houses can let you Save food you buy.
- Units get tired and need sleep
- Units can starve
- Units can create a business and become an employer
    - Units can Buy and Sell Food, as well as gift (add Generosity and Hustle Factors) - Possibly integrity
    - Units can sell Nothing of Value (add Cunningness and Naivity)
- Units can create food and sell it to other Units
- Food and Businesses can have a 'quality' or 'tier' 
- Units can 'mate' which bonds them together, and multiplies utility && expenses
- Child Units cost money 
- Implement the above in procedural ways to allow the 'society' to evolve via the genetic algorithm
- Food and Work sources need to be depletable to enforce dynamic behaviours
- Limit the number of workers to an employer
    - Determine what an employer status looks like, as well as potential business factors
- Add potential for sickness, disease and death
    - Malnutrition
    - Depression
    - Sexually Transmitted Diseases
    - Conflict
    - Exhaustion
- Implementing a Navigation Function for Turning and Velocity
    - Allow Children to Inherit Parent Traits\
- Create a Throttle mechanism (GUI) for changing parameters in real time.