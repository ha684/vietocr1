from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
import os
config = Cfg.load_config_from_file(r'D:\Workspace\python_code\vietocr1\config\vgg-transformer.yml')
dataset_params = {
    'name': 'ha2',
    'data_root': r'D:\Data\New folder\data',
    'train_annotation': 'label_train.txt',
    'valid_annotation': 'label_test.txt',
}

params = {
    'print_every': 1,
    'valid_every': 1000,
    'iters': 100000,
    'checkpoint': './checkpoint/transformers_checkpoint.pth',
    'export': './weights/transformers.pth',
    'metrics': 10000,
    'patience': 5
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda'
config['dataloader']['num_workers'] = 0
config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&''()*+,-./:;<=>?@[\]^_`{|}~²≥—–“”³…' + ' ' + '́' + "'"
trainer = Trainer(config, pretrained=False)

checkpoint_path = './checkpoint/vgg_transformer_checkpoint.pth'
if os.path.exists('/kaggle/input/vgg_transformer1/pytorch/default/2/vgg_transformer_checkpoint.pth'):
    print("Checkpoint found. Resuming training...")
    del config['optimizer']['max_lr']
    del config['optimizer']['pct_start']
    config['optimizer']['init_lr'] = 0.000001
    config['optimizer']['n_warmup_steps'] = 10
    trainer.load_checkpoint('/kaggle/input/vgg_transformer1/pytorch/default/2/vgg_transformer_checkpoint.pth')
else:
    print("No checkpoint found. Starting training from scratch...")

try:
    trainer.train()
    trainer.save_checkpoint(checkpoint_path)
except KeyboardInterrupt:
    trainer.save_checkpoint(checkpoint_path)
    print("Training interrupted. Checkpoint saved.")

config.save('config1.yaml')
