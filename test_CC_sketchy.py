import torch
from dataset.sketchy_dataset import TestSkDataset, TestPhDataset
from model.encoder_sketchy import Encoder
from model.cross_interaction import CrossInteraction
from model.self_interaction import SelfInteraction
from model.simple_match import SimpleMatch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.ckpt_utils import *
from utils.learning_utils import *
from utils.loss import cal_triplet_loss


def train():
    torch.cuda.manual_seed(1000)
    torch.manual_seed(1000)

    # load model
    ph_encoder = Encoder("mid").cuda()
    sk_encoder = Encoder("mid").cuda()

    self_interaction = SelfInteraction(0.5)
    match = CrossInteraction("2norm")
    # match=SimpleMatch('2norm')

    ckpt_path = "/home/xjq/code/shoe/DLI-Net/results/checkpoint/2021-11-20_11:20:55.237224/ckpt-best.pth"
    ckpt = torch.load(ckpt_path)
    ph_encoder.load_state_dict(ckpt["ph_encoder"])
    sk_encoder.load_state_dict(ckpt["sk_encoder"])

    ph_test_root = "/home/xjq/code/dataset/sketchy/photo/tx_000100000000"
    ph_test_txt = "/home/xjq/code/dataset/sketchy/photo_unseen.txt"
    sk_test_root = "/home/xjq/code/dataset/sketchy/sketch/tx_000100000000"
    sk_test_txt = "/home/xjq/code/dataset/sketchy/sketch_unseen.txt"
    test_ph_dataset = TestPhDataset(ph_test_root, ph_test_txt)
    test_sketch_dataset = TestSkDataset(sk_test_root, sk_test_txt)

    test_ph_loader = DataLoader(
        test_ph_dataset, batch_size=32, shuffle=False, num_workers=8
    )
    test_sk_loader = DataLoader(
        test_sketch_dataset, batch_size=32, shuffle=False, num_workers=8
    )

    # test
    acc_1, acc_5, acc_10 = get_retrieval_acc(
        ph_encoder,
        sk_encoder,
        test_ph_loader,
        test_sk_loader,
        0,
        self_interaction,
        match=match,
        dataset="sketchy",
    )

    print("acc@1:{:.4f}, acc@5:{:.4f}, acc@10:{:.4f}".format(acc_1, acc_5, acc_10))


if __name__ == "__main__":
    train()
