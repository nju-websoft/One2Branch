# coding=utf-8
from mindnlp.transformers import T5Tokenizer
from models.modeling_t5_branching import T5ForConditionalGeneration
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops
import mindspore.context as context
from tqdm import trange
import random
from utils import save_dataset, set_seed, save_model
import json
import argparse
import time
import copy
from eval_script_msqa import evaluate_msqa
from read_datasets import *
import ast
import numpy as np

def cat_answers(answers):
    return split_symbol.join(answers)

def parsing(text):
    return text.split(split_symbol)


def branching_labels(answer_ids):
    common_ancestor = []
    for i, answer in enumerate(answer_ids):
        common_ancestor.append([j for j in range(len(answer_ids))])
        # common_ancestor.append([i])
    answer_num = np.size(answer_ids, axis=0)
    seq_len = np.size(answer_ids, axis=1)
    labels = [[] for i in range(answer_num)]
    common_num = [[] for i in range(answer_num)]
    for seq_i in range(seq_len):
        for a_i, ancestor in enumerate(common_ancestor):
            for ancestor_i in ancestor:
                vob_id = answer_ids[ancestor_i][seq_i]
                if vob_id != -100:
                    if len(labels[a_i]) != seq_i + 1:
                        labels[a_i].append([vob_id])
                    else:
                        if vob_id not in labels[a_i][seq_i]:
                            labels[a_i][seq_i].append(vob_id)
                else:
                    break
        for a_i, ancestor in enumerate(copy.deepcopy(common_ancestor)):
            vob_id1 = answer_ids[a_i][seq_i]
            for ancestor_i in ancestor:
                if a_i == ancestor_i:
                    continue
                vob_id2 = answer_ids[ancestor_i][seq_i]
                if vob_id1 != vob_id2:
                    common_ancestor[a_i].remove(ancestor_i)
        for a_i, ancestor in enumerate(common_ancestor):
            vob_id = answer_ids[a_i][seq_i]
            if vob_id != -100:
                common_num[a_i].append(len(ancestor))
    return labels, common_num

def get_input_feature_train(features, tokenizer, max_length, max_target_len):
    input_list, decoder_input_ids = [], []
    max_answer_len = 0
    features_new = []
    decoder_num = []

    for b_i, sample in enumerate(features):
        answers = copy.deepcopy(sample['answers'])
        assert len(answers) > 0
        question = sample['question']
        if use_context:
            context = sample['context']
            input_list.append(f'Question: {question} Context: {context}')
        else:
            input_list.append(f'Question: {question}')

    for group_i, sample in enumerate(features):
        answers = copy.deepcopy(sample['answers'])
        assert len(answers) > 0
        # if len(answers) == 0:
        #     continue
        negatives = []
        if 'pred' in sample:
            answers_norm = [ans.lower() for ans in answers]
            pred_ans = sample['pred']
            for pred in pred_ans:
                if pred.lower() not in answers_norm:
                    negatives.append(pred)

        if len(negatives) > 1:
            negatives = negatives[:1]
            # negatives = random.sample(negatives, 1)

        encoding = tokenizer(answers + negatives,
                             padding='longest',
                             max_length=max_target_len,
                             truncation=True)
        answer_ids = encoding.input_ids
        answer_ids = [
            [label if label != tokenizer.pad_token_id else -100 for label in labels_example] for labels_example in
            answer_ids
        ]
        negative_ids = answer_ids[len(answers):]
        answer_ids = answer_ids[: len(answers)]
        labels, common_nums = branching_labels(answer_ids)
        assert len(labels) == len(answers)
        for a_i, (answer_id, label, common_num) in enumerate(zip(answer_ids, labels, common_nums)):
            sample_new = copy.deepcopy(sample)
            if len(label) > max_answer_len:
                max_answer_len = len(label)
            sample_new['label'] = label
            answer_id_new = []
            for item in answer_id:
                if item != -100:
                    answer_id_new.append(item)
            sample_new['decoder_input_id'] = answer_id_new
            label_mask = []
            for c_num in common_num:
                # label_mask.append(1 / c_num)
                label_mask.append(1)
            sample_new['label_mask'] = label_mask
            features_new.append(sample_new)

        def prefix_len(list1, list2):
            idx = 0
            while len(list1) < idx and len(list2) < idx and list1[idx] == list2[idx]:
                idx += 1
            return idx

        assert len(negatives) == len(negative_ids)
        for negative_id in negative_ids:
            sample_new = copy.deepcopy(sample)
            answer_id_new = []
            for item in negative_id:
                if item != -100:
                    answer_id_new.append(item)
            sample_new['decoder_input_id'] = answer_id_new
            if len(answer_id_new) > max_answer_len:
                max_answer_len = len(answer_id_new)
            max_prefix_len = 0
            for answer_id in answer_ids:
                max_prefix_len = max(max_prefix_len, prefix_len(negative_id, answer_id))
            sample_new['label_mask'] = [0] * max_prefix_len + [1] * (len(negative_id) - max_prefix_len)
            sample_new['label'] = [[]] * len(negative_id)
            features_new.append(sample_new)
        decoder_num.append(len(answers) + len(negatives))
    features = features_new
    labels = np.zeros([len(features), max_answer_len, vocab_size])
    label_masks = np.zeros([len(features), max_answer_len])
    answers_list = []
    for b_i, sample in enumerate(features):
        question = sample['question']
        answers_list.append(sample['answers'])
        decoder_input_id = copy.deepcopy(sample['decoder_input_id'])
        while len(decoder_input_id) < max_answer_len:
            decoder_input_id.append(-100)

        decoder_input_ids.append(decoder_input_id)
        label_mask = sample['label_mask']
        label = sample['label']
        assert len(label) == len(label_mask)
        for seq_i, (seq_label, m) in enumerate(zip(label, label_mask)):
            for l in seq_label:
                labels[b_i][seq_i][l] = 1
            label_masks[b_i][seq_i] = m

    input_ids, input_masks = tokenizer_fun(tokenizer, input_list, max_length)


    input_ids = Tensor(input_ids, mindspore.int32)
    input_masks = Tensor(input_masks, mindspore.int32)
    labels = Tensor(labels, mindspore.int32)
    decoder_input_ids = Tensor(decoder_input_ids, mindspore.int32)
    label_masks = Tensor(label_masks, mindspore.int32)


    return input_ids, input_masks, decoder_input_ids, labels, label_masks, decoder_num



def tokenizer_fun(tokenizer, input_texts, max_len):
    input_ids = []
    for input_text in input_texts:
        ids = tokenizer.encode(input_text)
        if len(ids) > max_len:
            ids = ids[:max_len]
        input_ids.append(ids)
    pad_id = tokenizer.pad_token_id
    max_len = max([len(input_id) for input_id in input_ids])
    attention_masks = []
    for input_id in input_ids:
        attention_mask = [1] * len(input_id) + [0] * (max_len - len(input_id))
        attention_masks.append(attention_mask)
        while len(input_id) < max_len:
            input_id.append(pad_id)
    return input_ids, attention_masks

def get_input_feature(features, tokenizer, max_length, max_target_len):
    input_list = []
    answers_list = []
    for sample in features:
        question = sample['question']
        if use_context:
            context = sample['context']
            input_list.append(f'Question: {question} Context: {context}')
        else:
            input_list.append(f'Question: {question}')
        answers_list.append(cat_answers(sample['answers']))




    input_ids, input_masks = tokenizer_fun(tokenizer, input_list, max_length)
    answer_ids, _ = tokenizer_fun(tokenizer, answers_list, max_target_len)

    answer_ids = [
        [label if label != tokenizer.pad_token_id else -100 for label in labels_example] for labels_example in
        answer_ids
    ]

    input_ids = Tensor(input_ids, mindspore.int32)
    input_masks = Tensor(input_masks, mindspore.int32)
    answer_ids = Tensor(answer_ids, mindspore.int32)

    return input_ids, input_masks, answer_ids


def evaluate(model, test_examples, eval_batch_size, tokenizer, max_len, max_target_len):
    model.set_train(False)
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    preds = {}
    golds = {}
    dataset_gold = []
    time_all = 0
    assert eval_batch_size == 1
    for step in step_trange:
        beg_index = step * eval_batch_size
        end_index = min((step + 1) * eval_batch_size, len(test_examples))
        batch_example = [example for example in test_examples[beg_index: end_index]]
        sample = batch_example[0]
        input_ids, input_masks, _ = get_input_feature(batch_example, tokenizer, max_len, max_target_len)
        beg = time.time()

        t5_output = model.generate(
            input_ids=input_ids,
            max_length=max_target_len,
            attention_mask=input_masks,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=False,
            branching_decoding=True,
            min_beam_num=args.min_beam_num,
            max_beam_num=args.max_beam_num
        )
        output_sequences = t5_output.sequences
        score_list = t5_output.score_list
        predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        assert len(predicts) == len(score_list)
        scores = []
        for score_item in score_list:
            scores.append(sum(score_item) / len(score_item))
        predicts = [(predict, score) for predict, score in zip(predicts, scores)]
        predicts = sorted(predicts, key=lambda x: x[1], reverse=True)
        predicts_new = []
        for predict in predicts:
            text, score = predict
            if score > 0:
                predicts_new.append(text)
        if len(predicts_new) == 0:
            predicts_new.append(predicts[0][0])
        spans_predicts = predicts_new

        if use_context:
            context = sample['context']
            spans_predicts_new = []
            for spans_predict in spans_predicts:
                if spans_predict.lower().strip() in context.lower():
                    spans_predicts_new.append(spans_predict)
            if len(spans_predicts_new) != 0:
                spans_predicts = spans_predicts_new

        end = time.time()
        time_all += (end - beg)
        for spans_predict, sample in zip(spans_predicts, batch_example):
            id = sample['id']
            answers = sample['answers']
            preds[id] = spans_predict
            golds[id] = answers
            dataset_gold.append({
                'id': id,
                'context': sample['context'],
                'answers': answers,
                'pred': spans_predict
            })
    print('Throughout:', round(len(test_examples) / time_all, 2))
    # print('time avg:', round(time_all/len(test_examples), 4))
    scores = evaluate_fun(copy.deepcopy(preds), copy.deepcopy(golds))
    return scores, preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default='t5-base',
                        type=str)
    parser.add_argument("--debug",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--only_eval",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--use_context",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--gpu",
                        default="1",
                        type=str)
    parser.add_argument("--dataset_name",
                        default='msqa',
                        type=str)
    parser.add_argument("--dataset_split",
                        default='in_house',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=24,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--ga',
                        type=int,
                        default=2,
                        help="Gradient accumulation")
    parser.add_argument("--results_save_path",
                        default='results',
                        type=str)
    parser.add_argument("--output_dir",
                        default='outputs',
                        type=str)
    parser.add_argument("--max_len",
                        default=2048,
                        type=int)
    parser.add_argument("--max_target_len",
                        default=512,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=40,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--acc_epoch",
                        default=-1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument("--min_beam_num",
                        default=1,
                        type=int)
    parser.add_argument("--max_beam_num",
                        default=20,
                        type=int)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    split_symbol = ' # '
    if args.model_name == 't5':
        args.model_name = '/data1/PTLM/t5_base/'
    elif args.model_name == 'unifiedqa':
        args.model_name = '/data1/PTLM/t5_unifiedqa_base/'
    only_eval = args.only_eval
    debug = args.debug
    model_name = args.model_name
    use_context = args.use_context
    read_dataset_fun = read_msqa
    evaluate_fun = evaluate_msqa
    data_path_base = f'../data/{args.dataset_split}/{args.dataset_name}/'
    data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/dev.json'
    data_path_test = f'{data_path_base}/test.json'

    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]

    prefix = 'One2Branch'
    if use_context:
        prefix += '_context'
    config_name = f'{prefix}/{args.dataset_name}/{model_name_abb}/{args.dataset_split}'

    parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}' \
                     f'_ga_{args.ga}'
    output_model_path = f'./{args.output_dir}/{config_name}/{parameter_name}/'
    path_save_result = f'./{args.results_save_path}/{config_name}/{parameter_name}/'
    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)
    if debug:
        train_examples = read_dataset_fun(data_path_train)[:10]
        dev_examples = read_dataset_fun(data_path_dev)[:10]
        test_examples = read_dataset_fun(data_path_test)[:10]
    else:
        train_examples = read_dataset_fun(data_path_train)
        dev_examples = read_dataset_fun(data_path_dev)
        test_examples = read_dataset_fun(data_path_test)


    train_batch_size = args.train_batch_size // args.ga
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)


    vocab_size = model.config.vocab_size
    print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                      "bs": args.train_batch_size,
                      'ga': args.ga,
                      "epoch": args.epoch_num,
                      'use_context': use_context,
                      "train_path": data_path_train,
                      "dev_path": data_path_dev,
                      "test_path": data_path_test,
                      "train_size": len(train_examples),
                      "train_examples": len(train_examples),
                      "dev_size": len(dev_examples),
                      "test_size": len(test_examples),
                      'max_len': args.max_len,
                      'output_model_path': output_model_path,
                      'path_save_result': path_save_result}, indent=2))



    if only_eval:
        scores, results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                       args.max_len, args.max_target_len)
        print('dev:', scores)
        save_dataset(path_save_result, '/dev.json', results_dev)

        scores, results_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer,
                                        args.max_len, args.max_target_len)
        print('test:', scores)
        save_dataset(path_save_result, '/test.json', results_test)
        exit(0)

    warm_up_ratio = 0.1
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0
    best_test_acc = 0
    best_dev_acc = 0
    best_dev_result, best_test_result = None, None

    for epoch in range(args.epoch_num):
        model.set_train(True)
        tr_loss, nb_tr_steps = 0, 0.1
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(args.seed + epoch)
        random.shuffle(order)
        step_count = len(train_examples) // train_batch_size
        if step_count * train_batch_size < len(train_examples):
            step_count += 1
        step_trange = trange(step_count)
        for step in step_trange:
            step_all += 1
            beg_index = step * train_batch_size
            end_index = min((step + 1) * train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch_example = [train_examples[index] for index in order_index]

            input_ids, input_masks, decoder_input_ids, labels, label_masks, decoder_num = \
                get_input_feature_train(batch_example, tokenizer, args.max_len, args.max_target_len)

            def forward_fn(input_ids, attention_mask, decoder_input_ids, labels, label_masks, decoder_num):
                output = model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  labels=decoder_input_ids,
                                  decoder_num=decoder_num,
                                  labels_branching=labels,
                                  return_dict=True,
                                  label_masks=label_masks)

                loss = output.loss
                return loss, None
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, _), grads = grad_fn(input_ids, input_masks, decoder_input_ids, labels, label_masks, decoder_num)
            optimizer(grads)
            tr_loss += loss.asnumpy()

            tr_loss += loss.item()
            nb_tr_steps += 1

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4))
            step_trange.set_postfix_str(loss_show)

        # if epoch >= 16:
        if epoch >= args.acc_epoch:
            scores_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size,
                                                                      tokenizer, args.max_len,args.max_target_len)
            print('dev:', scores_dev)
            scores = sum([scores_dev[key] for key in scores_dev.keys()])
            if scores > best_dev_acc:
                best_dev_acc = scores
                print('new best')

    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)

