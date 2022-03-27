from torch.utils.data.dataloader import DataLoader
from model.self_interaction import SelfInteraction
from model.cross_interaction import CrossInteraction
from model.simple_match import SimpleMatch
from utils.learning_utils import get_retrieval_acc
from dataset.sketchy_dataset import TestPhDataset,TestSkDataset
import torch
from torch.utils.data import DataLoader

from model.encoder_sketchy import Encoder
# ph_encoder=Encoder(feature_type='global').cuda()
# sk_encoder=Encoder(feature_type='global').cuda()

ph_encoder=Encoder().cuda()
sk_encoder=Encoder().cuda()
self_interaction=SelfInteraction(0.5)
match=CrossInteraction('2norm')
# self_interaction = None
# match = SimpleMatch('2norm')
ckpt=torch.load('/home/xjq/code/shoe/DLI-Net/ckpt/dli-sketchy.pth')
ph_encoder.load_state_dict(ckpt['ph_encoder'])
sk_encoder.load_state_dict(ckpt['sk_encoder'])

ph_test_root='/home/xjq/code/dataset/sketchy/photo/tx_000000000000'
ph_test_txt='/home/xjq/code/dataset/sketchy/photo_test_relative_path.txt'
sk_test_root='/home/xjq/code/dataset/sketchy/sketch/tx_000100000000'
sk_test_txt='/home/xjq/code/dataset/sketchy/sketch_test_relative_path.txt'
test_ph_dataset=TestPhDataset(ph_test_root,ph_test_txt)
test_sk_dataset=TestSkDataset(sk_test_root,sk_test_txt)
test_ph_loader=DataLoader(test_ph_dataset,batch_size=32,shuffle=False,num_workers=8)
test_sk_loader=DataLoader(test_sk_dataset,batch_size=32,shuffle=False,num_workers=8)

acc_1, acc_5, acc_10 = get_retrieval_acc(
            ph_encoder,
            sk_encoder,
            test_ph_loader,
            test_sk_loader,
            0,
            self_interaction,
            match,
            dataset='sketchy'
        )

print('acc@1:{:.4f}, acc@5:{:.4f}, acc@10:{:.4f}'.format(acc_1,acc_5,acc_10))