from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
import os

config = Cfg.load_config_from_file("./vietocr/config/vit_seq2seq.yml")
dataset_params = {
    "name": "ha",
    "data_root": r"D:\Workspace\python_code\ImageGenerations\images_out",
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
    "batch_size": 1,
}
config["vocab"] = (
    'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'
    "()*+,-./:;<=>?@[\]^_`{|}~’ " + "'"
)
config["trainer"].update(params)
config["dataset"].update(dataset_params)
config["device"] = "cuda:0"
config["dataloader"]["num_workers"] = 0


trainer = Trainer(config)

checkpoint_path = "./checkpoint/vgg_transformer_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print("Checkpoint found. Resuming training...")
    trainer.load_checkpoint(checkpoint_path)
else:
    print("No checkpoint found. Starting training from scratch...")

try:
    trainer.train()
    trainer.save_checkpoint(checkpoint_path)
except KeyboardInterrupt:
    pass
    trainer.save_checkpoint(checkpoint_path)
    print("Training interrupted. Checkpoint saved.")

config.save("config1.yaml")
