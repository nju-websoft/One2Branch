# coding=utf-8

from transformers import get_linear_schedule_with_warmup, T5Tokenizer
from transformers import T5ForConditionalGeneration

from tqdm import trange
import random
from utils import save_dataset, set_seed, save_model
import json
import argparse
import time
import copy
from eval_scripts.eval_script_msqa import evaluate_msqa
from read_datasets import *
import ast
import torch

device = torch.device("cuda:0")

def cat_answers(answers):
    return split_symbol.join(answers)

def parsing(text):
    return text.split(split_symbol)


def get_input_feature(features, tokenizer, max_length):
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

    def tokenizer_fun(input_ids, max_len):
        encoding = tokenizer(input_ids,
                             padding='longest',
                             max_length=max_len,
                             truncation=True)
        ids = encoding.input_ids
        mask = encoding.attention_mask
        return ids, mask
    input_ids, input_masks = tokenizer_fun(input_list, max_length)
    answer_ids, _ = tokenizer_fun(answers_list, max_length)

    answer_ids = [
        [label if label != tokenizer.pad_token_id else -100 for label in labels_example] for labels_example in
        answer_ids
    ]

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_masks = torch.tensor(input_masks, dtype=torch.long).to(device)
    answer_ids = torch.tensor(answer_ids, dtype=torch.long).to(device)

    return input_ids, input_masks, answer_ids


@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, tokenizer, max_len, max_target_len):
    model.eval()
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
        input_ids, input_masks, _ = get_input_feature(batch_example, tokenizer, max_len)
        beg = time.time()

        t5_output = model.generate(
            input_ids=input_ids,
            max_length=max_target_len,
            attention_mask=input_masks,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        output_sequences = t5_output.sequences
        predicts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        spans_predicts = [parsing(predict) for predict in predicts]

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
    print('time avg:', round(time_all/len(test_examples), 4))
    scores = evaluate_fun(copy.deepcopy(preds), copy.deepcopy(golds))
    return scores, preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        default='/data1/PTLM/t5_base/',
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
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--init",
                        default=None,
                        type=ast.literal_eval,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
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
    data_path_base = f'./data/{args.dataset_split}/{args.dataset_name}/'
    data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/dev.json'
    data_path_test = f'{data_path_base}/test.json'

    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]

    prefix = 'One2Seq'
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

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    n_gpu = torch.cuda.device_count()
    layer_num = model.config.num_layers
    layer_per_gpu = layer_num // n_gpu
    layer_per_gpu_remainder = layer_num % n_gpu
    device_map = {}
    cur_layer = 0
    for n in range(n_gpu):
        device_map[n] = []
        if n < layer_per_gpu_remainder:
            layer_assigned = layer_per_gpu + 1
        else:
            layer_assigned = layer_per_gpu

        for i in range(layer_assigned):
            device_map[n].append(cur_layer)
            cur_layer += 1
    model.parallelize(device_map)

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
                      'path_save_result': path_save_result,
                      'init_checkpoint': args.init_checkpoint}, indent=2))
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    if only_eval:
        args.init = True

    if args.init and args.init_checkpoint is None:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', args.init_checkpoint)
    elif args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', args.init_checkpoint)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    t_total = args.epoch_num * (len(train_examples) // train_batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=int(warm_up_ratio * (t_total)),
                                                num_training_steps=t_total)
    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0
    best_test_acc = 0
    best_dev_acc = 0
    best_dev_result, best_test_result = None, None
    if args.init_checkpoint is not None:
        scores_dev, results_dev, readable_results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                                 args.max_len, args.max_target_len)
        scores = sum([scores_dev[key] for key in scores_dev.keys()])
        print('scores_dev:', scores_dev)
        best_dev_acc = scores

    for epoch in range(args.epoch_num):
        tr_loss, nb_tr_steps = 0, 0.1
        early_stop += 1
        order = list(range(len(train_examples)))
        random.seed(args.seed + epoch)
        random.shuffle(order)
        model.train()
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
            input_ids, input_masks, labels = get_input_feature(batch_example, tokenizer, args.max_len)
            output = model(input_ids=input_ids, attention_mask=input_masks, labels=labels)
            loss = output.loss
            # loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss = loss / args.ga
            loss.backward()
            if (step + 1) % args.ga == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(
                round(tr_loss / nb_tr_steps, 4)) + f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
            step_trange.set_postfix_str(loss_show)

        # if epoch >= 16:
        if epoch >= args.acc_epoch:
            scores_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size,
                                                                      tokenizer, args.max_len,args.max_target_len)
            print('dev:', scores_dev)
            scores = sum([scores_dev[key] for key in scores_dev.keys()])
            if scores > best_dev_acc:
                best_dev_acc = scores
                print('save new best')
                save_model(output_model_path, model, optimizer)

    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)

    ###############################

    init_checkpoint = f'{output_model_path}/pytorch_model.bin'
    checkpoint = torch.load(init_checkpoint, map_location='cpu')
    model_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_dict, False)
    print('init from:', init_checkpoint)
    scores, results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer, args.max_len, args.max_target_len)
    print('dev:', scores)
    save_dataset(path_save_result, '/dev.json', results_dev)

    scores, results_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer, args.max_len, args.max_target_len)
    print('test:', scores)
    save_dataset(path_save_result, '/test.json', results_test)

