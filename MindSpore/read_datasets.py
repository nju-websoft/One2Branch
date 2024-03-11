
from utils import read_dataset
from eval_script_msqa import get_entities

from os.path import join
import os

def read_keyphrase(path_src, path_trg):
    dataset_src = read_dataset(path_src)
    dataset_trg = read_dataset(path_trg)
    dataset = []
    for sample_src, sample_trg in zip(dataset_src, dataset_trg):
        if sample_src.strip() == "":
            continue
        if len(sample_trg.strip().split(';')) == "":
            continue
        dataset.append({
            'source': sample_src.strip(),
            'target': sample_trg.strip()
        })
    return dataset

def read_KPTimes(path):
    dataset_init = read_dataset(path)
    dataset = []
    for sample in dataset_init:
        title = sample['title'].strip()
        abstract = sample['abstract'].strip()

        dataset.append({
            'source': title + ' <eos> ' + abstract,
            'target': sample['keyword']
        })
    return dataset

def read_msqa(path):
    dataset_init = read_dataset(path)
    dataset = []
    for sample in dataset_init:
        if 'label' not in sample:
            dataset = dataset_init
            break
        id = sample['id']
        question = sample['question']
        context = sample['context']
        label = sample['label']
        answers = get_entities(label, context)

        answers_super = []
        offset = 2
        for ans, beg, end in answers:
            beg_s, end_s = beg, end
            if beg >= offset:
                beg_s -= offset
            if end <= len(context)-offset:
                end_s += offset
            answers_super.append(' '.join(context[beg_s: end_s]))

        answers_extract = [answer[0] for answer in answers]
        dataset.append(
            {
                'id': id,
                'context': ' '.join(context),
                'question': ' '.join(question),
                'answers': answers_extract,
                'answers_super': answers_super
            }
        )
    return dataset


def read_nq(path):
    dataset_init = read_dataset(path)
    dataset = []
    for sample in dataset_init:
        id = sample['id']
        question = sample['question']
        ctxs = sample['ctxs']
        ctxs = [item['text'] for item in ctxs]
        dataset.append(
            {
                'id': id,
                'context': ctxs,
                'question': question,
                'answers': sample['answers']
            }
        )
    return dataset

def read_cmqa(path):
    dataset_init = read_dataset(path)
    dataset = []
    for sample in dataset_init:
        id = sample['id']
        question = sample['question']
        context = sample['context']
        answers = sample['coarse'] + sample['fine']
        answers = sorted(answers, key=lambda x: x[1][0])
        answers = [answer[0] for answer in answers]
        dataset.append(
            {
                'id': id,
                'context': context,
                'question': question,
                'answers': answers
            }
        )
    return dataset

def read_arc_da(path):
    dataset_init = read_dataset(path)
    dataset = []
    for sample in dataset_init:
        dataset.append(
            {
                'id': sample['question_id'],
                'question': sample['question'],
                'answers': sample['answers']
            }
        )
    return dataset


def read_squad2(path):
    dataset = read_dataset(path)
    dataset_new = []
    for sample in dataset:
        paragraphs = sample['paragraphs']
        for p_samples in paragraphs:
            context = p_samples['context']
            qas = p_samples['qas']
            for qa_sample in qas:
                question = qa_sample['question']
                id = qa_sample['id']
                if 'answers' in qa_sample:
                    answers = qa_sample['answers']
                else:
                    answers = [
                        {
                            "text": context.split(' ')[0],
                            "answer_start": context.index(context.split(' ')[0])
                        }
                    ]
                # answers = sorted(answers, key=lambda x: x['answer_start'])
                answers_idx = []
                answers_text = []

                answers_super = []
                offset = 2
                for answer_item in answers:
                    text = answer_item['text']
                    answer_start = answer_item['answer_start']
                    answer_end = answer_start + len(text)
                    answers_idx.append([answer_start, answer_end])
                    # print('context:',context)
                    # print('text:',text)
                    # print(context[answer_start: answer_end])
                    # print('-----')
                    # assert context[answer_start: answer_end] == text
                    answers_text.append(text)

                    beg_s, end_s = answer_start, answer_end
                    offset_label = 0
                    for i in range(1, 100):
                        if beg_s == 0:
                            break
                        if context[beg_s - 1] == ' ':
                            offset_label += 1
                        if offset_label == offset + 1:
                            break
                        beg_s = beg_s - 1

                    for i in range(1, 100):
                        if end_s >= len(context) - 1:
                            end_s = len(context)
                            break
                        end_s = end_s + 1
                        if context[end_s] == ' ':
                            offset_label += 1
                        if offset_label == offset + 1:
                            break
                    answers_super.append(context[beg_s: end_s])


                # if len(answers_text) > 1:
                #     print('answers_text:', len(answers_text))
                #     print(len(list(set(answers_text))))
                # answers_text = list(set(answers_text))
                # if len(answers_text) > 1:
                #     print(len(answers_text))
                dataset_new.append({
                    'id': id,
                    'question': question,
                    'context': context,
                    'answers': answers_text,
                    'answers_super': answers_super
                    # 'answers_idx': answers_idx
                })
    return dataset_new


def read_relations():
    """
    Loads hierarchy file and returns set of relations
    """
    # relations = set([])
    # singeltons = set([])
    ancestors = {}
    with open('data/official/bgc/hierarchy.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            rel = line.split('\t')
            if len(rel) > 1:
                # rel = (rel[0], rel[1])
                ancestors[rel[1]] = rel[0]
    #         else:
    #             singeltons.add(rel[0][:-1])
    #             continue
    #         relations.add(rel)
    # print(singeltons)
    return ancestors


def read_bgc(path):
    """
    Loads labels and blurbs of dataset
    """
    from bs4 import BeautifulSoup
    dataset = []
    ancestors = read_relations()
    data = []
    soup = BeautifulSoup(open(join(path), 'rt').read(), "html.parser")
    for id, book in enumerate(soup.findAll('book')):
        categories = set()
        answers = []
        book_soup = BeautifulSoup(str(book), "html.parser")
        for t in book_soup.findAll('topics'):
            s1 = BeautifulSoup(str(t), "html.parser")
            structure = ['d3', 'd2', 'd1', 'd0']
            # assert s1.findAll('d0') == 1
            for level in structure:
                for t1 in s1.findAll(level):
                    node = str(t1.string)
                    if node in categories:
                        continue
                    categories.add(node)
                    path_nodes = []
                    path_nodes.append(node)
                    while node in ancestors:
                        node = ancestors[node]
                        path_nodes.append(node)
                        categories.add(node)
                    path_nodes = [node for node in reversed(path_nodes)]
                    path_nodes = ' # '.join(path_nodes)
                    answers.append(path_nodes)

                    # flag = True
                    # for answer in answers:
                    #     if path_nodes in answer:
                    #         flag = False
                    #         break
                    # if flag:
        # print((str(book_soup.find("body").string), categories))
        # break
        # data.append((str(book_soup.find("body").string), categories))

        # a_set = set()
        # for answer in answers:
        #     answer = answer.split(' # ')
        #     for a in answer:
        #         a_set.add(a)
        #
        # if len(a_set) != len(categories):
        #     print('a_set:', a_set)
        #     print('categories:', categories)
        #     print('answers:', answers)
        #
        # assert len(a_set) == len(categories)

        dataset.append({
            'id': str(id),
            'context': str(book_soup.find("body").string),
            'answers': answers,
        })
    # answers_count = 0
    # max_count = 0
    # for sample in dataset:
    #     answers = sample['answers']
    #     max_count = max(max_count, len(sample['context'].split(' ')))
    #     answers_count += len(answers)
    # print('max_count:', max_count)
    return dataset

if __name__ == '__main__':
    # read_bgc('./data/official/bgc/train.txt')
    # read_relations()
    dataset = read_KPTimes('data/KPTimes/test.jsonl')
    print('dataset:', len(dataset))