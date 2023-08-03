import utils.config as config
from models.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config.save_memory:
    # xxxx8888
    # enable_sliced_attention()
    pass