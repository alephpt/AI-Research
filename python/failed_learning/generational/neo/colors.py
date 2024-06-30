import random

# Primary Colors for Individuals are Blue, Red and Yellow.. 
Blue = (0, 0, 255)
Red = (255, 0, 0)
Yellow = (255, 255, 0)

# Seconary Colors are Purple, Green and Orange
Purple = (255, 0, 255)          # Offspring of Blue and Red
Green = (0, 255, 0)             # Offspring of Blue and Yellow
Orange = (255, 165, 0)          # Offspring of Red and Yellow

# Territiary Colors for 'cross-breeding'
Olive = (128, 128, 0)            # Offspring of Green and Orange or Yellow and Purple
Russet = (128, 70, 27)           # Offspring of Orange and Purple
Brown = (139, 69, 19)            # Offspring of Purple and Green

colorMap = {
    "Blue": Blue,
    "Red": Red,
    "Yellow": Yellow,
    "Purple": Purple,
    "Green": Green,
    "Orange": Orange,
    "Olive": Olive,
    "Russet": Russet,
    "Brown": Brown
}

def criticalRaceTheory(parentA, parentB):
    # If the parents are the same color, the offspring will be the same color
    if parentA == parentB:
        return parentA
    
    if parentA == "Blue" and parentB == "Red" or parentA == "Red" and parentB == "Blue":
        return "Purple"
    elif parentA == "Blue" and parentB == "Yellow" or parentA == "Yellow" and parentB == "Blue":
        return "Green"
    elif parentA == "Red" and parentB == "Yellow" or parentA == "Yellow" and parentB == "Red":
        return "Orange"
    elif parentA == "Green" and parentB == "Orange" or parentA == "Orange" and parentB == "Green":
        return "Olive"
    elif parentA == "Orange" and parentB == "Purple" or parentA == "Purple" and parentB == "Orange":
        return "Russet"
    elif parentA == "Purple" and parentB == "Green" or parentA == "Green" and parentB == "Purple" \
        or parentA == "Yellow" and parentB == "Purple" or parentA == "Purple" and parentB == "Yellow" \
        or parentA == "Red" and parentB == "Green" or parentA == "Green" and parentB == "Red" \
        or parentA == "Blue" and parentB == "Orange" or parentA == "Orange" and parentB == "Blue":
        return "Brown"
    else:
        return random.choice([parentA, parentB])
    

if __name__ == "__main__":
    random.seed()
    parentA = random.choice(list(colorMap.keys()))
    print("Parent A: ", parentA)
    
    parentB = random.choice(list(colorMap.keys()))
    print("Parent B: ", parentB)
    
    offspring = criticalRaceTheory(parentA, parentB)
    print("Offspring: ", offspring)
        
    
    