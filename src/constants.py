import torch

# Path or parameters for DATA_DIR
DATA_DIR = 'DATA_DIR'
SP_DIR = f'{DATA_DIR}/sp'
SRC_DIR = ''
TRG_DIR = 'trg'
SRC_RAW_DATA_NAME = 'raw_data.src'
TRG_RAW_DATA_NAME = 'raw_data.trg'
TRAIN_NAME = 'train.txt'
VALID_NAME = 'valid.txt'
TEST_NAME = 'test.txt'

# Parameters for sentencepiece tokenizer
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
src_model_prefix = 'src_sp'
trg_model_prefix = 'trg_sp'
sp_vocab_size = 16000
character_coverage = 1.0
model_type = 'unigram'

# Parameters for Transformer & training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 1e-4
batch_size = 25
seq_len = 200
num_heads = 8
num_layers = 5
d_model = 512
d_ff = 2048
d_k = d_model // num_heads
drop_out_rate = 0.1
num_epochs = 4
beam_size = 8
ckpt_dir = 'saved_model'
