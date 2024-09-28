from vietocr.model.trainer import Trainer
from vietocr.tool.config import Cfg
import os
config = Cfg.load_config_from_file('./vietocr/config/vit_seq2seq.yml')
dataset_params = {
    'name': 'ha1',
    'data_root': '/kaggle/input/folder5',
    'train_annotation': 'label5/label_train.txt',
    'valid_annotation': 'label5/label_test.txt',
    'train_gen' : 'train_ha4',
    'valid_gen' : 'valid_ha4',
    'image_height': 32
}

params = {
    'print_every': 100,
    'valid_every': 2000,
    'epochs' : 10,
    'checkpoint': './checkpoint/seq2seq_checkpoint.pth',
    'export': './weights/seq2seq.pth',
    'metrics': 1733846,
    'patience': 5,
    'batch_size': 32,
}
config['vocab'] = '!"$%&()*+,-./0123456789:;<=>?@ABCDEFGHIJ�KLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}²³¼½¾ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ '
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda'
config['dataloader']['num_workers'] = 2
trainer = Trainer(config, pretrained=False)

checkpoint_path = './checkpoint/seq2seq_checkpoint.pth'
if os.path.exists('/kaggle/input/vitseq2seq1/pytorch/default/1/seq2seq_checkpoint.pth'):
    print("Checkpoint found. Resuming training...")
    trainer.load_checkpoint('/kaggle/input/vitseq2seq1/pytorch/default/1/seq2seq_checkpoint.pth')
else:
    print("No checkpoint found. Starting training from scratch...")

try:
    trainer.train()
    trainer.save_checkpoint(checkpoint_path)
except KeyboardInterrupt:
    trainer.save_checkpoint(checkpoint_path)
    print("Training interrupted. Checkpoint saved.")

config.save('config1.yaml')
