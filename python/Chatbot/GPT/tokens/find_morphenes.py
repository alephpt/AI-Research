import re, os

# open all the files in training_data
for file in os.listdir("../training_data"):
    with open("../training_data/" + file, "r", encoding="utf8") as f:
        strings = f.read().splitlines()

unique_words = set(strings)

# open dedup_tokens.txt
with open("dedup_tokens.txt", "r") as f:
    tokens = f.read().splitlines()

print("Unique Words: " + str(len(unique_words)))
print("Tokens: " + str(len(tokens)))

# iterate through words in strings
for line in strings:
    words = line.split(' ')
    
    for word in words:
        word = re.sub(r'[^a-zA-Z]', '', word)
        ##print("Checking \"" + word + "\" with length " + str(len(word)))
        # if word is in tokens, skip it
        if word in tokens:
            continue

        # check if the word contains any morphemes
        for token in tokens:
            if len(token) > len(word) or len(token) == 1:
                continue
            
            finder = word.find(token)
            if finder != -1:
                #print("Found \"" + token + "\" in \"" + word + "\" at index " + str(finder))

                # if it does, split the word into subwords and ignore the token
                subwords = word.split(token)
                
                # for each subword, check if it is in tokens
                for subword in subwords:
                    if not subword:
                        continue
                    
                    #print("\tChecking sub \"" + subword + "\" with length " + str(len(subword)))
                    # if it is, skip it
                    if subword in tokens:
                        #print("\t\t\"" + subword + "\" is in tokens.. skipping")
                        continue

                    # if it is not, check if it contains any morphemes
                    for token in tokens:
                        if len(token) > len(subword) or len(token) == 1:
                            continue
                        
                        # if it does, split the word into sub-subwords
                        finder = subword.find(token)
                        if finder != -1:
                            #print("\t\tFound \"" + token + "\" in \"" + subword + "\" at index " + str(finder))
                            # for each sub-subword,
                            subsubwords = subword.split(token)
                            
                            # check if it is in tokens
                            for subsubword in subsubwords:
                                if not subsubword:
                                    continue
                                
                                #print("\t\t\tChecking subsub \"" + subsubword + "\" with length " + str(len(subsubword)))
                                # if it is, skip it
                                if subsubword in tokens:
                                    #print("\t\t\t\t\"" + subsubword + "\" is in tokens.. skipping")
                                    offset = subword.split(subsubword)
                                    for subsubsubword in offset:
                                        if not subsubsubword:
                                            continue
                                        
                                        #print("\t\t\t\tChecking subsubsub \"" + subsubsubword + "\" with length " + str(len(subsubsubword)))
                                        if subsubsubword in tokens:
                                            #print("\t\t\t\t\t\"" + subsubsubword + "\" is in tokens.. skipping")
                                            continue
                                        else:
                                            print("\t\t\t\t\t\"" + subsubsubword + "\" is not in tokens.. adding")
                                            tokens.append(subsubsubword)
                                            
                                            # check the other part of the subword
                                            offset = subword.split(subsubsubword)
                                            for subsubsubsubword in offset:
                                                if not subsubsubsubword or subsubsubsubword == subsubsubword or len(subsubsubsubword) == 1:
                                                    continue
                                                
                                                if not subsubsubsubword in tokens:
                                                    print("\t\t\t\t\t\"" + subsubsubsubword + "\" is not in tokens.. adding")
                                                    tokens.append(subsubsubsubword)
                                    continue
                                # if it is not, add it to tokens
                                else: 
                                    print("\t\t\t\t\"" + subsubword + "\" is not in tokens.. adding")
                                    tokens.append(subsubword)
                        
                    # do a post check to see if the subword is in tokens
                    # iterate through all the tokens
                    for token in tokens:
                        if len(token) > len(subword) or len(token) == 1:
                            continue
                        
                        subword_found = False
                        
                        # check if we can find a token in the subword
                        finder = subword.find(token)
                        if finder != -1:
                            subword_found = True
                            break
                        
                    # if we can't, add the subword to tokens
                    if not subword_found:
                        print("\t\t\"" + subword + "\" is not in tokens.. adding")
                        tokens.append(subword)
                        
        # do a post check to see if the word is in tokens
        # iterate through all the tokens
        for token in tokens:
            if len(token) > len(word) or len(token) == 1:
                continue
            
            word_found = False
            
            # check if we can find a token in the word
            finder = word.find(token)
            if finder != -1:
                word_found = True
                break
            
        # if we can't, add the word to tokens
        if not word_found:
            print("\"" + word + "\" is not in tokens.. adding")
            tokens.append(word)
            
print("Tokens: " + str(len(tokens)))

# save new sorted token list
with open("max_tokens.txt", "w") as f:
    tokens = list(set(tokens))
    tokens.sort(key=len, reverse=True)
    f.write("\n".join(tokens))