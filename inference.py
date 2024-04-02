import numpy as np

# Example usage
mapping_path = "./model/mapping_indo.csv"
mapping = {'\t': 0,
 '\n': 1,
 ' ': 2,
 '_': 3,
 'a': 4,
 'b': 5,
 'c': 6,
 'd': 7,
 'e': 8,
 'f': 9,
 'g': 10,
 'h': 11,
 'i': 12,
 'j': 13,
 'k': 14,
 'l': 15,
 'm': 16,
 'n': 17,
 'o': 18,
 'p': 19,
 'q': 20,
 'r': 21,
 's': 22,
 't': 23,
 'u': 24,
 'v': 25,
 'w': 26,
 'x': 27,
 'y': 28,
 'z': 29,
 '\xa0': 30,
 '²': 31,
 'â': 32,
 'î': 33,
 'ï': 34,
 'ð': 35,
 'ô': 36,
 'ù': 37,
 'ú': 38,
 'û': 39}

# Example usage
reverse_mapping_path = "./model/reverse_mapping_indo.csv"
reverse_mapping = {0: '\t',
 1: '\n',
 2: ' ',
 3: '_',
 4: 'a',
 5: 'b',
 6: 'c',
 7: 'd',
 8: 'e',
 9: 'f',
 10: 'g',
 11: 'h',
 12: 'i',
 13: 'j',
 14: 'k',
 15: 'l',
 16: 'm',
 17: 'n',
 18: 'o',
 19: 'p',
 20: 'q',
 21: 'r',
 22: 's',
 23: 't',
 24: 'u',
 25: 'v',
 26: 'w',
 27: 'x',
 28: 'y',
 29: 'z',
 30: '\xa0',
 31: '²',
 32: 'â',
 33: 'î',
 34: 'ï',
 35: 'ð',
 36: 'ô',
 37: 'ù',
 38: 'ú',
 39: 'û'}



# Function to generate lyrics
# The function to generate text from model
def Lyrics_Generator(starter,Ch_count, model_load,L_symb=40):
    generated= ""
    starter = starter.lower() 
    seed=[mapping[char] for char in starter]
    generated += starter 
    # Generating new text of given length
    for i in range(Ch_count):
        seed=[mapping[char] for char in starter]
        x_pred = np.reshape(seed, (1, len(seed), 1))
        x_pred = x_pred/ float(L_symb)
        prediction = model_load.predict(x_pred, verbose=0)[0]
        
        # Getting the index of the next most probable index
        prediction = np.asarray(prediction).astype('float64')
        prediction = np.log(prediction) / 1.0 
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        # probas = np.random.multinomial(1, prediction, 1)
        index = np.argmax(prediction)
        next_char = reverse_mapping[index]  
        
        # Generating new text
        generated += next_char
        starter = starter[1:] + next_char
       
    return generated