import glob
import logging
import time
import argparse

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer

from tools.common import init_logger
from models.bert_for_classifier import BertSoftmaxForClassifier
from tools.train_helper import *


# 初始化参数
MODEL_CLASSES = {
    'bert': (BertConfig, BertSoftmaxForClassifier, BertTokenizer),
}
BERT_BASE_DIR = 'prev_trained_model/albert_chinese_small'
DATA_DIR = 'datasets'
OUTPUT_DIR = 'outputs'
TASK_NAME = 'ccb'

args = argparse.Namespace()
args.model_type = 'bert'
args.model_name_or_path = BERT_BASE_DIR
args.task_name = TASK_NAME
args.do_train = True
args.do_lower_case = True
args.data_dir = '{}/{}'.format(DATA_DIR, TASK_NAME)
args.train_max_seq_length = 512
args.eval_max_seq_length = 512
args.batch_size = 24
args.learning_rate = 3e-5
args.num_train_epochs = 4
args.output_dir = '{}/{}_output/'.format(OUTPUT_DIR, TASK_NAME)
args.overwrite_output_dir = True
args.seed = 42
args.do_predict = True
# 选择最好的checkpoint
args.predict_checkpoints = 4
args.overwrite_cache = False
args.gradient_accumulation_steps = 1
args.weight_decay = 0.01
args.warmup_proportion = 0.1
args.adam_epsilon = 1e-8
args.max_grad_norm = 1.0
args.loss_type = 'focal'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
time_ = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
if os.path.exists(args.output_dir) and os.listdir(
        args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError(
        'Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.'.format(
            args.output_dir))

# Setup CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.n_gpu = torch.cuda.device_count()
args.device = device
logger.warning('Device: %s, n_gpu: %s', device, args.n_gpu)

# Set seed
seed_everything(args.seed)

# Prepare ccb task
args.task_name = args.task_name.lower()
if args.task_name not in processors:
    raise ValueError('Task not found: %s' % args.task_name)
processor = processors[args.task_name]()
label_list = processor.get_labels()
args.id2label = {i: label for i, label in enumerate(label_list)}
args.label2id = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

# Load pretrained model and tokenizer
args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
config.loss_type = args.loss_type
tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
model = model_class.from_pretrained(args.model_name_or_path, config=config)
model.to(args.device)

logger.info('Training/evaluation parameters %s', args)
# Training
if args.do_train:
    train_dataset = load_and_cache_examples(args, tokenizer, data_type='train')
    train(args, train_dataset, model, tokenizer)

# Prediction
if args.do_predict:
    checkpoints = []
    if args.predict_checkpoints > 0:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger('transformers.modeling_utils').setLevel(logging.WARN)  # Reduce logging
        checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
    logger.info('Predict the following checkpoints: %s', checkpoints)
    for checkpoint in checkpoints:
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ''
        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(args.device)
        predict(args, model, tokenizer, prefix=prefix)
