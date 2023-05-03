# open on_the_shortness_of_life from ../training_data/
strings = []
with open("../training_data/on_the_shortness_of_life.txt", "r") as f:
    strings = f.readlines()

# open dedup_tokens.txt
tokens = []

with open("dedup_tokens.txt", "r") as f:
    tokens = f.readlines()

# iterate through words in strings
for word in strings:
    print(word)
    # if word is in tokens, skip it

    # check if the word contains any morphemes
        # if it does, split the word into subwords and ignore the token
        # for each subword, check if it is in tokens
            # if it is, skip it
        # if it is not, check if it contains any morphemes
            # if it does, split the word into sub-subwords
                # for each sub-subword,
                # check if it is in tokens
                    # if it is, skip it
                    # if it is not, add it to tokens
            # if it does not, add it to tokens
        # if it does not, add it to tokens
