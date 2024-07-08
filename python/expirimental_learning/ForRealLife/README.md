# Creating The Matrix
## Deep Genetic Learning Outline

## Goal
 - Determine if an Agent can figure out how to survive, if the only two options are Eating and working - and the own two indicators of state are intiailly energy and money.
 - Once an agent is able to survive, they should learn how to reproduce.
 - Reproduction introduces the concept of a relationship, influence or attraction
 - At this point we can scale work/eating into 'trade' with a deals 'cost':'return' being determined by their level of influence.
 - At this point we can scale the agent to be able to create their own markets and homes base on their money + food and money + energy, respectively.
 - We want to start with one, and allow for the agent to explore options of planting food genetically to see if they make a correlation as a society, or on accident.

### Concept:
  - Can we simulate society without telling the agent how to exist within it?
  - Can we define rules for a society 'member' to exist within?
  - What sort of free agents would emerge in society e.g. 
        what would be varrying 'optimal' ways of "scoring high"?

Visit the [Outline](https://github.com/alephpt/AI-Research/blob/main/python/expirimental_learning/ForRealLife/outline.md) for the full writeup. (subject to change)

Brief:
### Rules:
- Agents require food to Survive
- Food costs 5 Money and provide 10 Energy
- Work costs 5 Energy and provide 10 Money
- Agents can reproduce with other Agents to create new Agents if M+F
- Agents can die if they run out of Energy
- Agents can be male or female
- Any Agent can be Mate but only Male and Female Agents can Reproduce, at an appropriated chance
- Mating brings happiness
- Reproducing brings a large happiness reward
- The 'Reinforcement' Q Network allows for agents to cross-breed, and potentially share information within a 'family'
- Epochs end when the entire population dies.
- Each Agent will have their own Table with trial and error, and the child will inherit a mix of their parents genes.
- The top happiness, wealth, energy and age, as well as the longest living generation are used to repopulate, with a shuffled mix of the worst performers

#### Deep Reinforcement Learning for Agents Behavior
   - Learn to Navigate
   - Learn to Survive
   - Learn to Reproduce
   - Reward Agents based on 'Happiness', second to 'Survival'
       - Reward = Lifespan + Happiness
   - Can we multi-sample async training buffers for 'higher resolution' ?

#### Generations of Agents
- Genetic Algorithm for Population Evolution
   -  Fill half of the population, with half of the furthest generation


Working Space:
## TODOs
 -[ ] Write paper on the concept of genetic reinforcement
 -[ ] Define Policies
 -[X] Add an 'Impulse Factor' to the Agent
 -[ ] Be able to 'hone' target
 -[ ] Be able to acheive interaction with a cell
 -[ ] Configure Table for State: Action determination
    -[ ] Integrate State as Key
    -[ ] Define Action Table
    -[ ] Determine Reward factor for for 'Best Action'
 

### Future Potential Implementations:
- Jobs provide Income for Some and Food for Others.. 
    by implementing a chance to trade, 
    and allowing them to pick one or the other : Vendor or Buyer
   - Vendor makes Money, and Buyer gets Food
        - Later, Banks can let you Save
        - Houses can let you Save food you buy.
- Agents get tired and need sleep
- Agents can starve
- Agents can create a business and become an employer
    - Agents can Buy and Sell Food, as well as gift (add Generosity and Hustle Factors) - Possibly integrity
    - Agents can sell Nothing of Value (add Cunningness and Naivity)
- Agents can create food and sell it to other Agents
- Food and Businesses can have a 'quality' or 'tier' 
- Agents can 'mate' which bonds them together, and multiplies utility && expenses
- Child Agents cost money 
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