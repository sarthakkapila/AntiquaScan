from CHARS import CHARS

def text_to_char_ids(text, char_to_id, max_length):                                                 
    char_ids = [char_to_id[char] if char in char_to_id else char_to_id['<unk>'] for char in text]
    padded_char_ids = char_ids[:max_length] + [char_to_id['<pad>']] * (max_length - len(char_ids))
    return padded_char_ids

char_to_id = {'<pad>': 0, '<unk>': 1} 
for char in CHARS:
    char_to_id[char] = len(char_to_id)


char_to_index = {char: index for index, char in enumerate(CHARS)}
index_to_char = {index: char for char, index in char_to_index.items()}

def tensor_to_text(tensor):
    text = ""
    for index in tensor:
        if index.item() == 0:  
            break
        text += index_to_char.get(index.item(), '?')  
    return text

max_length = 25

input_size = len(CHARS)
output_size = len(CHARS)+1