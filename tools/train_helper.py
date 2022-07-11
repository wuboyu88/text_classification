import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup, AdamW

from callback.progressbar import ProgressBar
from tools.common import seed_everything, logger
from metrics.cls_metrics import EvaluateScore
from processors.cls_seq import convert_examples_to_features
from processors.cls_seq import text_processors as processors
from processors.cls_seq import collate_fn


def load_and_cache_examples(args, tokenizer, data_type='train'):
    processor = processors[args.task_name]()
    # Load data features from cache or datasets file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        data_type,
        args.model_name_or_path.split('/')[-1],
        args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info('Loading features from cached file %s', cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info('Creating features from datasets file at %s', args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train'
                                                else args.eval_max_seq_length,
                                                cls_token_at_end=False,
                                                pad_on_left=False,
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )

        logger.info('Saving features into cached file %s', cached_features_file)
        torch.save(features, cached_features_file)
    # Convert to Tensors and build datasets
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    datasets = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return datasets


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_batch_size = args.batch_size * max(1, args.n_gpu)
    # todo 打乱顺序
    train_sampler = RandomSampler(train_dataset)
    # todo 不同batch的sequence length可以不同https://www.zhihu.com/question/439438113
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size,
                                  collate_fn=collate_fn, generator=torch.Generator().manual_seed(args.seed))

    # len(train_dataloader)表示每个epoch中batch的个数
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    # todo Adamw 即 Adam + weight decay ,效果与 Adam + L2正则化相同,但是计算效率更高,因为L2正则化需要在loss中加入正则项,之后再算梯度,
    # todo 最后在反向传播,而Adamw直接将正则项的梯度加入反向传播的公式中,省去了手动在loss中加正则项这一步
    # todo https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay
    # todo bias不需要正则化
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0},
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # todo warmup https://www.jianshu.com/p/1867381de345
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Instantaneous batch size per GPU = %d', args.batch_size)
    logger.info('  Total train batch size (w. parallel, distributed & accumulation) = %d',
                train_batch_size * args.gradient_accumulation_steps)
    logger.info('  Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', t_total)

    global_step = 0
    train_loss = 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=args.num_train_epochs)
    train_results = dict()
    dev_results = dict()
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch + 1)
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            pbar(step, {'loss': loss.item()})
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # todo 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        # todo 每个epoch评估模型效果并保存模型
        # Only evaluate when single GPU otherwise metrics may not average well
        train_result = evaluate(args, model, tokenizer, data_type='train')
        dev_result = evaluate(args, model, tokenizer, data_type='dev')
        # evaluate(args, model, tokenizer, data_type='test')

        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch + 1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info('Saving model checkpoint to %s', output_dir)
        tokenizer.save_vocabulary(output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        logger.info('Saving optimizer and scheduler states to %s', output_dir)

        train_result = {'{}_{}'.format(epoch + 1, k): v for k, v in train_result.items()}
        train_results.update(train_result)
        dev_result = {'{}_{}'.format(epoch + 1, k): v for k, v in dev_result.items()}
        dev_results.update(dev_result)

        logger.info('\n')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    with open(os.path.join(args.output_dir, 'train_results.txt'), 'w') as writer:
        for key in sorted(train_results.keys()):
            writer.write('{} = {}\n'.format(key, str(train_results[key])))

    with open(os.path.join(args.output_dir, 'dev_results.txt'), 'w') as writer:
        for key in sorted(dev_results.keys()):
            writer.write('{} = {}\n'.format(key, str(dev_results[key])))
    return train_loss / global_step


def evaluate(args, model, tokenizer, data_type='train', prefix=''):
    metric = EvaluateScore()
    eval_dataset = load_and_cache_examples(args, tokenizer, data_type=data_type)
    eval_batch_size = args.batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,
                                 collate_fn=collate_fn, generator=torch.Generator().manual_seed(args.seed))
    # Evaluation!
    logger.info('***** Running evaluation %s *****', prefix)
    logger.info('  Num examples = %d', len(eval_dataset))
    logger.info('  Batch size = %d', eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        label_ids = inputs['labels'].reshape(-1).cpu().numpy().tolist()
        labels = [args.id2label[ele] for ele in label_ids]
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        preds = [args.id2label[ele] for ele in pred_ids]
        metric.update(preds=preds, labels=labels)
        pbar(step)
    eval_loss = eval_loss / nb_eval_steps
    total_info, class_info = metric.result()
    results = total_info
    results['loss'] = eval_loss
    logger.info('***** {} total results {} *****'.format(data_type, prefix))
    info = '-'.join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info('***** {} class results {} *****'.format(data_type, prefix))
    for key in sorted(class_info.keys()):
        logger.info('******* %s results ********' % key)
        info = '-'.join([f' {key}: {value:.4f} ' for key, value in class_info[key].items()])
        logger.info(info)
    logger.info('\n')
    return results


def predict(args, model, tokenizer, prefix=''):
    test_dataset = load_and_cache_examples(args, tokenizer, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn,
                                 generator=torch.Generator().manual_seed(args.seed))
    # Predict!
    logger.info('***** Running prediction %s *****', prefix)
    logger.info('  Num examples = %d', len(test_dataset))
    logger.info('  Batch size = %d', 1)
    results = []
    output_predict_file = os.path.join(args.output_dir, prefix, 'test_prediction.json')
    pbar = ProgressBar(n_total=len(test_dataloader), desc='Predicting')

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': None}
            outputs = model(**inputs)
            logits = outputs[0]
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds = [args.id2label[ele] for ele in pred_ids]
        json_d = dict()
        json_d['id'] = step
        json_d['pred'] = ' '.join(preds)
        results.append(json_d)
        pbar(step)
    logger.info('\n')
    with open(output_predict_file, 'w') as writer:
        for record in results:
            writer.write(json.dumps(record, ensure_ascii=False) + '\n')
