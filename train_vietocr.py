import os
import torch
import torch.multiprocessing as mp
import numpy as np
from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg

def main_worker(local_rank, config):
    # Initialize DDP and set device
    torch.distributed.init_process_group(backend='nccl', init_method='env://',world_size=2, rank=local_rank)
    torch.cuda.set_device(local_rank)

    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    print(f"Process {local_rank} initialized on device {torch.cuda.current_device()}")
    trainer = Trainer(config, local_rank=local_rank)

    checkpoint_path = "./checkpoint/seq2seq_checkpoint.pth"
    if local_rank == 0 and os.path.exists('/kaggle/input/vitseq2seq1/pytorch/default/1/seq2seq_checkpoint.pth'):
        print("Checkpoint found. Resuming training...")
        trainer.load_checkpoint('/kaggle/input/vitseq2seq1/pytorch/default/1/seq2seq_checkpoint.pth')
    else:
        if local_rank == 0:
            print("No checkpoint found. Starting training from scratch...")

    try:
        trainer.train()
        if local_rank == 0:
            trainer.save_checkpoint(checkpoint_path)
    except KeyboardInterrupt:
        if local_rank == 0:
            trainer.save_checkpoint(checkpoint_path)
            print("Training interrupted. Checkpoint saved.")

    if local_rank == 0:
        config.save("config1.yaml")

if __name__ == '__main__':
    config = Cfg.load_config_from_file("./vietocr/config/vit_seq2seq.yml")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    dataset_params = {
        "name": "ha",
        "data_root": "/kaggle/input/folder1/label1",
        "train_annotation": "label_train.txt",
        "valid_annotation": "label_test.txt",
    }

    params = {
        "print_every": 1,
        "valid_every": 1000,
        "iters": 100000,
        "epochs": 10,
        "checkpoint": "./checkpoint/transformers_checkpoint.pth",
        "export": "./weights/transformers.pth",
        "metrics": 10000,
        "patience": 5,
        "batch_size": 8,
    }

    config["vocab"] = (
        'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬ'
        'bBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆ'
        'fFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌ'
        'ôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢ'
        'pPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰ'
        'vVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~’ '
        "'"
    )

    config["trainer"].update(params)
    config["dataset"].update(dataset_params)
    # Remove config["device"] as devices are set per process
    config["dataloader"]["num_workers"] = 0

    num_gpus = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=num_gpus, args=(config,))
