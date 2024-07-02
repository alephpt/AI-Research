


### Outline

Concept:
  - Can we simulate society without telling the agent how to exist within it?
  - Can we define rules for a society 'member' to exist within?
  - What sort of free agents would emerge in society e.g. 
        what would be varrying 'optimal' ways of "scoring high"?

Rules:
- Agents require food to Survive
- Food costs 5 Money and provide 10 Energy
- Work costs 5 Energy and provide 10 Money
- Agents can reproduce with other Agents to create new Agents if M+F
- Agents can die if they run out of Energy
- Agents can be male or female

- Deep Reinforcement Learning for Agents Behavior
   - Learn to Navigate
   - Learn to Survive
   - Learn to Reproduce
   - Reward Agents based on 'Happiness', second to 'Survival'
       - Reward = Lifespan + Happiness

Generations of Agents
- Genetic Algorithm for Population Evolution
   -  Fill half of the population, with half of the furthest generation

Future Potential Implementations:
- Jobs provide Income for Some and Food for Others.. 
    by implementing a chance to trade, 
    and allowing them to pick one or the other : Vendor or Buyer
   - Vendor makes Money, and Buyer gets Food
        - Later, Banks can let you Save
        - Houses can let you Save food you buy.
- Agents get tired and need sleep
- Agents can starve
- Agents can create a business and become an employer
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
    - Allow Children to Inherit Parent Traits