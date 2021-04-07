from ast import literal_eval


def get_seq2str(seq):
    with open("data/int2str.txt", "r", encoding="utf-8") as fil:
        dict_ = fil.read()

    seq2str = literal_eval(dict_)

    return [seq2str[i] for i in seq]


def mapping_seq(preds):
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().tolist()
    
    for num, pred in enumerate(preds):
        memory = -1
        temp = []
        for i in pred:
            if i == 0:
                memory = -1
            elif i != memory:
                temp.append(i)
                memory = 1
        preds[num] = temp
    
    return preds
