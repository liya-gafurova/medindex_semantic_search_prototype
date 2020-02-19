from nltk import  word_tokenize

passage_tokens = word_tokenize("Participants dependence.###Key exclusion criteria were intracranial haemorrhage from\
###See full tableTable 1. Baseline otherwise")

best_span = [9, 10]

# Text concatination:
#  -- [paragraph1][splitter][paragraph2][splitter][paragraph3]   '[splitter]'.join(paragraphs)
#
# BiDAF parses , i.e. '###' like ['#', '#', '#']
# -- passage_tokens looks like ['Participants', 'dependence.', '#', '#', '#', 'Key', 'exclusion', 'criteria' ...]
#
# Parser:
#  -- get splitter combination position
#  -- get paragraph position between splitter combination
#  -- check whether best_span is within one passage. If not, raise an exeption

def get_paragraph_positons(passage_tokens: list, splitter: str) :
    print(passage_tokens)
    for i, pas in enumerate(passage_tokens):
        print('{} -- {}'.format(i, pas))
    splitter_len = len(splitter)
    passsage_tokens_len = len(passage_tokens)
    splitter_positions = [(-1,-1)]
    passage_positions = []
    for i in range(passsage_tokens_len):
        # проверка на разделитель, сранивать шаблоны
        substr = ''.join(passage_tokens[i:i+splitter_len])
        if substr== splitter:
            splitter_positions.append((i, i+2))

    splitter_positions.append((passsage_tokens_len, passsage_tokens_len))
    for idx in range(1, len(splitter_positions)):
        passage_positions.append((splitter_positions[idx-1][1]+1, splitter_positions[idx][0]-1))


    print('{}'.format(splitter_positions))
    return passage_positions

def locate_passage_with_answer(passage_tokens: list, best_span: list, splitter: str):
    # gives positions of paragraph with answer (best_span)
    passage_positions = get_paragraph_positons(passage_tokens, splitter)
    for paragraph_pos in passage_positions:
        if paragraph_pos[0] and best_span[1]<= paragraph_pos[1] :
            return paragraph_pos

    raise Exception('best_span is NOT within one paragraph')

paragraph_with_answer_position = locate_passage_with_answer(passage_tokens=passage_tokens,best_span= best_span, splitter='###')
print(' '.join(passage_tokens[paragraph_with_answer_position[0]:paragraph_with_answer_position[1]+1]))