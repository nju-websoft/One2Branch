# coding=utf-8

from transformers import get_linear_schedule_with_warmup, T5Tokenizer
from models.modeling_t5_branching import T5ForConditionalGeneration
from tqdm import trange
import random
from utils import save_dataset, set_seed, save_model
import json
import argparse
import time
import copy
from tqdm import tqdm
from eval_scripts.eval_script_keyphrase import keyphrase_evaluate
from read_datasets import *
import ast
import numpy as np
import torch

device = torch.device("cuda:0")
@torch.no_grad()
def evaluate(model, test_examples, eval_batch_size, tokenizer, max_len, max_target_len):
    model.eval()
    step_count = len(test_examples) // eval_batch_size
    if step_count * eval_batch_size < len(test_examples):
        step_count += 1
    time_all = 0
    sources, targets, predictions = [], [], []
    assert eval_batch_size == 1
    for sample in tqdm(test_examples):
        source = sample['source']
        sources.append(source)
        target = sample['target']
        targets.append(target)
        input_ids, input_masks = get_input_feature_test([sample], tokenizer, max_len)
        beg = time.time()

        t5_output = model.generate(
            input_ids=input_ids,
            # encoder_outputs=ModelOutput(last_hidden_state=encoder_hidden),
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
        predicts = [item.replace(special_token_to, special_token_from) for item in predicts]
        assert len(predicts) == len(score_list)
        scores = []
        for score_item in score_list:
            scores.append(sum(score_item) / len(score_item))

        predicts = [(predict, score) for predict, score in zip(predicts, scores)]
        predicts = sorted(predicts, key=lambda x: x[1], reverse=True)
        if only_eval_train is False:
            predicts_new = []
            for predict in predicts:
                text, score = predict
                if score > 0:
                    predicts_new.append(text)
            if len(predicts_new) == 0:
                predicts_new.append(predicts[0][0])
            spans_predicts = predicts_new
        else:
            predicts_new = [item[0] for item in predicts]
            spans_predicts = predicts_new

        end = time.time()
        time_all += (end - beg)
        spans_predict_str = ';'.join(spans_predicts).lower()
        sample['pred'] = spans_predict_str
        predictions.append(spans_predict_str)

    print('Throughout:', round(len(test_examples) / time_all, 2))
    results = keyphrase_evaluate(copy.deepcopy(sources), copy.deepcopy(targets),
                                 copy.deepcopy(predictions))

    macro_avg_f1_5_present = results['macro_avg_f1@5_present']
    macro_avg_f1_M_present = results['macro_avg_f1@M_present']
    macro_avg_f1_5_absent = results['macro_avg_f1@5_absent']
    macro_avg_f1_M_absent = results['macro_avg_f1@M_absent']

    scores = {
        'present@5': macro_avg_f1_5_present,
        'present@M': macro_avg_f1_M_present,
        'absent:5': macro_avg_f1_5_absent,
        'absent:M': macro_avg_f1_M_absent,
        'sum': macro_avg_f1_5_present + macro_avg_f1_M_present + macro_avg_f1_5_absent + macro_avg_f1_M_absent
    }
    return scores, predictions

def tokenizer_fun(tokenizer, input_ids, max_len):
    encoding = tokenizer(input_ids,
                         padding='longest',
                         max_length=max_len,
                         truncation=True)
    ids = encoding.input_ids
    mask = encoding.attention_mask
    return ids, mask

def get_input_feature_train(features, tokenizer, max_length):
    input_list, decoder_input_ids = [], []
    max_answer_len = 0
    features_new = []
    decoder_num = []
    for group_i, sample in enumerate(features):
        target = sample['target'].strip()
        target = target.replace(special_token_from, special_token_to)
        answers = target.split(';')
        if '<peos>' in answers:
            answers.remove('<peos>')
        assert '<peos>' not in answers
        answers = list(set(answers))
        if len(answers) > 20:
            answers = answers[:20]
        if len(answers) == 0:
            continue

        source = sample['source'].strip()
        source = source.replace(special_token_from, special_token_to)
        if '<eos>' in source:
            sp = source.split('<eos>')
            title, context = sp
            input_list.append(f'Title: {title} Context: {context}')
        else:
            input_list.append(f'Context: {source}')

        negatives = []
        if 'pred' in sample:
            answers_norm = [ans.lower() for ans in answers]
            pred_ans = sample['pred'].split(';')
            for pred in pred_ans:
                if pred.lower() not in answers_norm:
                    negatives.append(pred)
        k = 1
        if len(negatives) > k:
            negatives = negatives[:k]
        encoding = tokenizer(answers + negatives,
                             padding='longest',
                             max_length=max_length,
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
                label_mask.append(1 / c_num)
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

    input_ids, input_masks = tokenizer_fun(tokenizer, input_list, max_length)
    assert len(decoder_num) == len(input_ids)
    features = features_new
    labels = np.zeros([len(features), max_answer_len, vocab_size])
    label_masks = np.zeros([len(features), max_answer_len])
    for b_i, sample in enumerate(features):
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
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_masks = torch.tensor(input_masks, dtype=torch.long).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(device)
    label_masks = torch.tensor(label_masks, dtype=torch.float).to(device)
    return input_ids, input_masks, decoder_input_ids, labels, label_masks, decoder_num


def get_input_feature_test(features, tokenizer, max_length):
    input_list = []
    for sample in features:
        source = sample['source'].strip()
        source = source.replace(special_token_from, special_token_to)
        if '<eos>' in source:
            sp = source.split('<eos>')
            title, context = sp
            input_list.append(f'Title: {title} Context: {context}')
        else:
            input_list.append(f'Context: {source}')

        if only_eval_train:
            args.min_beam_num = len(sample['target'].strip()) + 2
            if args.min_beam_num > 20:
                args.min_beam_num = 20

    input_ids, input_masks = tokenizer_fun(tokenizer, input_list, max_length)
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_masks = torch.tensor(input_masks, dtype=torch.long).to(device)
    return input_ids, input_masks


def branching_labels(answer_ids):
    common_ancestor = []
    for i, answer in enumerate(answer_ids):
        common_ancestor.append([j for j in range(len(answer_ids))])
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
    parser.add_argument("--only_eval_train",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--gpu",
                        default="1",
                        type=str)
    parser.add_argument("--dataset_name",
                        default='kptimes',
                        type=str)
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--ga',
                        type=int,
                        default=16,
                        help="Gradient accumulation")
    parser.add_argument("--results_save_path",
                        default='results',
                        type=str)
    parser.add_argument("--output_dir",
                        default='outputs',
                        type=str)
    parser.add_argument("--init",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--use_context",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument("--save_model",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--max_len",
                        default=512,
                        type=int)
    parser.add_argument("--max_target_len",
                        default=60,
                        type=int)
    parser.add_argument("--lr",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num",
                        default=20,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--acc_epoch",
                        default=-1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--use_negative",
                        default=False,
                        type=ast.literal_eval)
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument("--min_beam_num",
                        default=10,
                        type=int)
    parser.add_argument("--max_beam_num",
                        default=20,
                        type=int)

    special_token_from = ''
    special_token_to = ''
    split_symbol = ' ; '
    args = parser.parse_args()
    gpu = args.gpu

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    print('gpu:', args.gpu)

    only_eval = args.only_eval
    only_eval_train = args.only_eval_train
    use_negative = args.use_negative
    debug = args.debug
    save_model_flag = args.save_model
    model_name = args.model_name
    use_context = args.use_context
    Tokenizer = T5Tokenizer

    dataset_name = args.dataset_name

    if 'kp20k' in dataset_name:
        special_token_from = '<digit>'
        special_token_to = '#'
        # read_fun = read_KPTimes
    # else:
    read_fun = read_dataset
    data_path_base = f'./data/{dataset_name}'
    if use_negative:
        # if 'large' in model_name and os.path.exists(f'{data_path_base}/train_pred_large_filter.json'):
        #     data_path_train = f'{data_path_base}/train_pred_large_filter.json'
        # else:
        data_path_train = f'{data_path_base}/train_pred.json'
    else:
        data_path_train = f'{data_path_base}/train.json'
    data_path_dev = f'{data_path_base}/dev.json'
    data_path_test = f'{data_path_base}/test.json'

    if args.model_name.endswith('/'):
        args.model_name = args.model_name[:-1]
    model_name_abb = args.model_name.split('/')[-1]

    config_name = f'{model_name_abb}/'
    if use_negative:
        parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}' \
                         f'_ga_{args.ga}_negative'
    else:
        parameter_name = f'lr_{args.lr}_seed_{args.seed}_bs_{args.train_batch_size}' \
                         f'_ga_{args.ga}'
    output_model_path = f'./One2Branch/{args.output_dir}/{dataset_name}/{config_name}/{parameter_name}/'
    path_save_result = f'./One2Branch/{args.results_save_path}/{dataset_name}/{config_name}/{parameter_name}/'
    os.makedirs(path_save_result, exist_ok=True)
    set_seed(args.seed)

    if use_negative and only_eval is False and args.init is False:
        args.init_checkpoint = output_model_path.replace('_negative', '') + '/pytorch_model.bin'

    if debug:
        train_examples = read_fun(data_path_train)[:10]
        dev_examples = read_fun(data_path_dev)[:10]
        test_examples = read_fun(data_path_test)[:10]
    else:
        train_examples = read_fun(data_path_train)
        dev_examples = read_fun(data_path_dev)
        test_examples = read_fun(data_path_test)
        if len(dev_examples) > 1000:
            dev_examples = dev_examples[:1000]

    train_batch_size = args.train_batch_size // args.ga
    tokenizer = Tokenizer.from_pretrained(args.model_name)
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
    print(device_map)
    model.parallelize(device_map)

    vocab_size = model.config.vocab_size
    print(json.dumps({"lr": args.lr, "model": args.model_name, "seed": args.seed,
                      "bs": args.train_batch_size,
                      'ga': args.ga,
                      "epoch": args.epoch_num,
                      'save_model': save_model_flag,
                      'min_beam_num': args.min_beam_num,
                      'max_beam_num': args.max_beam_num,
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
                      'only_eval_train': only_eval_train,
                      'init_checkpoint': args.init_checkpoint}, indent=2))
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    if only_eval or only_eval_train:
        args.init = True

    if args.init and args.init_checkpoint is None:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'

        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', init_checkpoint)

    elif args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', args.init_checkpoint)

    if only_eval_train:
        scores, results_train = evaluate(model, train_examples, args.eval_batch_size, tokenizer,
                                         args.max_len, args.max_target_len)
        print(f'train:', scores)
        train_save_name = f'train_pred.json'
        save_dataset(data_path_base, train_save_name, train_examples)
        exit(0)

    if only_eval:
        scores_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                           args.max_len, args.max_target_len)
        print('scores_dev:', scores_dev)
        save_dataset(path_save_result, f'/dev_{args.min_beam_num}_{args.max_beam_num}.json', results_dev)

        scores, results_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer,
                                       args.max_len, args.max_target_len)
        print(f'test:', scores)
        save_dataset(path_save_result, f'/test_{args.min_beam_num}_{args.max_beam_num}.json', results_test)

        for test_name in ['inspec', 'krapivin', 'nus', 'semeval']:
            data_path_test = f'./data/{test_name}/test.json'
            test_examples = read_fun(data_path_test)
            scores, results_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer,
                                                     args.max_len, args.max_target_len)
            print(f'{test_name}:', scores)
            save_dataset(path_save_result, f'/{test_name}.json', results_test)
        exit(0)

    warm_up_ratio = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    t_total = args.epoch_num * (len(train_examples) // train_batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=1000,
                                                num_training_steps=t_total)
    step_count, step_all, early_stop = 0, 0, 0
    best_dev_rouge_score, best_test_rouge_score = 0, 0
    best_test_acc = 0
    best_dev_acc = 0
    best_dev_result, best_test_result = None, None
    if args.init_checkpoint is not None or args.init:
        scores_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                                 args.max_len, args.max_target_len)
        print(path_save_result)
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
            input_ids, input_masks, decoder_input_ids, labels, label_masks, decoder_num =\
                get_input_feature_train(batch_example, tokenizer, args.max_len)

            t5_output = model(
                input_ids=input_ids,
                attention_mask=input_masks,
                labels=decoder_input_ids,
                decoder_num=decoder_num,
                labels_branching=labels,
                return_dict=True,
                label_masks=label_masks)
            loss = t5_output.loss
            loss = loss.mean()

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

        if epoch > args.acc_epoch:
            scores_dev, results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                                        args.max_len, args.max_target_len)
            print('dev:', scores_dev)
            scores = sum([scores_dev[key] for key in scores_dev.keys()])
            if scores >= best_dev_acc:
                best_dev_acc = scores
                print('save new best')
                if save_model_flag:
                    save_model(output_model_path, model, optimizer)
                    save_dataset(path_save_result, '/dev.json', results_dev)
                else:
                    save_dataset(path_save_result, '/dev.json', results_dev)

                    scores_test, results_test = evaluate(model, test_examples, args.eval_batch_size,
                                                                           tokenizer,
                                                                           args.max_len, args.max_target_len)
                    print('test:', scores_test)
                    save_dataset(path_save_result, '/test.json', results_test)

    print('best_dev_result:', best_dev_result)
    print('best_test_result:', best_test_result)
    print(path_save_result)

    ###############################
    if save_model_flag:
        init_checkpoint = f'{output_model_path}/pytorch_model.bin'
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.load_state_dict(model_dict, False)
        print('init from:', init_checkpoint)
        scores, results_dev = evaluate(model, dev_examples, args.eval_batch_size, tokenizer,
                                       args.max_len, args.max_target_len)
        print('dev:', scores)
        save_dataset(path_save_result, '/dev.json', results_dev)

        scores, results_test = evaluate(model, test_examples, args.eval_batch_size, tokenizer,
                                        args.max_len, args.max_target_len)
        print('test:', scores)
        save_dataset(path_save_result, '/test.json', results_test)
