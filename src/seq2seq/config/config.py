from os.path import dirname, abspath, join
import sys
import warnings

warnings.filterwarnings('ignore', message='foo bar')

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '../'))
print(CODE_DIR)
sys.path.append(CODE_DIR)

TEST_PATH = abspath(join(CODE_DIR, 'data', 'test.csv'))
DATA_PATH = abspath(join(CODE_DIR, 'data', 'categorical_data.csv'))
WEIGHTS_DIR = abspath(join(CODE_DIR, 'trained_model', 'seq2seq.h5'))
INPUT_TOKENIZER_DIR = abspath(join(CODE_DIR, 'trained_model', 'tokenizer.json'))
OUTPUt_TOKENIZER_DIR = abspath(join(CODE_DIR, 'trained_model', 'output_tokenizer.json'))
EMBEDDING_DIR = abspath(join(CODE_DIR, 'trained_model', 'Datagram_w2v_with_target.text'))

CONF_DIR = abspath(join(THIS_DIR, '../', 'trained_model','models','config.json'))
MODEL_DIR = abspath(join(THIS_DIR, '../', 'trained_model','models'))
ID2LABEL_PATH = abspath(join(THIS_DIR, 'id2label.json'))
LABEL2ID_PATH = abspath(join(THIS_DIR, 'label2id.json'))
PCLASS2CATEGORY = abspath(join(CODE_DIR, 'config', 'PclassClass2category_id.json'))
CLASS2CATEGORY = abspath(join(CODE_DIR, 'config', 'class2category_id.json'))
# config
BATCH_SIZE = 512
EPOCHS = 2
LATENT_DIM = 1024
LATENT_DIM_DECODER = 1024
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 60000
EMBEDDING_DIM = 512