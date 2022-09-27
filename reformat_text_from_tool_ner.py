import json
from underthesea import word_tokenize

data = []
with open('all.jsonl', encoding='utf-8') as f:
    for line in f:
        a = line
        data.append(json.loads(a))

def extract_text(dic):
    list_id_sentence_error = []
    rs = []
    #dic = data[3]
    sentence = dic["text"]
    sentence.replace("\/", " ")
    #sentence.replace(")", " ")
    if (sentence[-1] == '.'):
        sentence = sentence[:-1]
    print(dic["id"])
    label_tagged = dic["entities"][0]['label']
    start = dic["entities"][0]['start_offset']
    end = dic["entities"][0]['end_offset']
    word_entities = sentence[start:end]


    list_words = sentence.split()

    # if list_words.index(word_entities.split()[-1]) >=0:
    if word_entities.split()[-1] in list_words  and  word_entities.split()[0] in list_words:
        lt = list_words.index(word_entities.split()[-1])
        st = list_words.index(word_entities.split()[0])
        if len(word_entities.split()) > 1:
            for word in list_words:
                _w = list_words.index(word)
                if word in word_entities and _w in range(st, lt + 1, 1):
                    label = label_tagged
                else:
                    label = 'O'
                s = {"word": word, "label": label}
                rs.append(s)
        else:
            for word in list_words:
                if word == word_entities:
                    label = label_tagged
                else:
                    label = 'O'
                s = {"word": word, "label": label}
                rs.append(s)
       # print(dic["id"])
    else:
        list_id_sentence_error.append(dic["id"])

    print(list_id_sentence_error)
    return rs


f = open("data_train.txt", "w", encoding='utf-8')

for list_items in data:
    _item = extract_text(list_items)
    for item in _item:
        f.write(item["word"] + " " + item["label"])
        f.write('\n')
    f.write('\n')

f.close()
