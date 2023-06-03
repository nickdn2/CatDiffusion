from dataclasses import dataclass
@dataclass
class TrainConfig():

    log_interval : int = 200
    eval_interval : int = 800
    checkpoint_interval : int = 1600
    epochs : int = 100
    lr : float = 10e-6
    seed : int = 40 #TODO Посмотреть, где надо будет зафиксировать рандом и реализовать
    lr_warmup_steps : int = 3400
    num_train_timesteps : int = 500
    board : bool = True
    device : str = 'cuda:0'
    keep_chekpoints : int = 3 #TODO Реализовать: Хранить последние k чекпоинтов
    checkpoint_dir : str = './logs/'
    log_dir : str = './logs/'
    cur_epoch : int = 0
    cur_iteration : int = 0
    eval_batch_size : int = 16

@dataclass
class DataConfig():

    all_in_memory : bool = False
    training_files : str = './datasets/cats/'
    train_shuffle : bool = True
    train_batch_size : int = 16 

@dataclass
class ModelConfig():

    sample_size : int = 128
    in_channels : int = 3
    out_channels : int = 3
    layers_per_block : int = 2
    block_out_channels : tuple = (128,128,256,256,512,512)
    down_block_types : tuple = (
        "DownBlock2D", 
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types :tuple = (
        "UpBlock2D",
        "AttnUpBlock2D", 
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )