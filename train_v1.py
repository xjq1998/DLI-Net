import torch
from dataset.qmul_v1_dataset import TrainDataset, TestPhDataset, TestSkDataset
from model.encoder import Encoder
from model.cross_interaction import CrossInteraction
from model.self_interaction import SelfInteraction
from model.simple_match import SimpleMatch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.ckpt_utils import *
from utils.learning_utils import *
from utils.loss import cal_triplet_loss
from utils.Logger import *
from options.train_options import TrainOptions
import itertools


def train():
    torch.cuda.manual_seed(1000)
    torch.manual_seed(1000)
    opt = TrainOptions().parse()
    logger = Logger(opt.logger_path).get_logger()
    logger.info("parameters: {}".format(opt))

    logger.info("loading model")
    ph_encoder = Encoder(opt.feature_type).cuda()
    sk_encoder = Encoder(opt.feature_type).cuda()

    self_interaction = None
    if opt.self_interaction:
        self_interaction = SelfInteraction(opt.k)
    if opt.cross_interaction:
        match = CrossInteraction(opt.norm_type)
    else:
        print("simple")
        match = SimpleMatch(opt.norm_type)

    lr = opt.lr
    epoch = 0
    best_acc = [0, 0, 0]

    logger.info("loading data")
    train_dataset = TrainDataset(opt.ph_train_root, opt.sk_train_root, opt.ph_train_txt)
    test_ph_dataset = TestPhDataset(opt.ph_test_root, opt.ph_test_txt)
    test_sketch_dataset = TestSkDataset(opt.sk_test_root, opt.sk_test_txt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    test_ph_loader = DataLoader(
        test_ph_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8
    )
    test_sk_loader = DataLoader(
        test_sketch_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8
    )

    # optimizer
    parameters = itertools.chain(ph_encoder.parameters(), sk_encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, betas=[0.9, 0.999])

    loss_tracker = AverageMeter()

    for i in range(epoch + 1, opt.epoch + 1):

        ph_encoder.train()
        sk_encoder.train()

        tqdm_batch = tqdm(train_loader, desc="Epoch-{} training".format(i))

        # t0 = time.time()
        for ph, sk in tqdm_batch:
            ph, sk = ph.cuda(), sk.cuda()
            ph_fea = ph_encoder(ph)
            sk_fea = sk_encoder(sk)
            if opt.self_interaction:
                ph_fea = self_interaction(ph_fea)
                sk_fea = self_interaction(sk_fea)

            loss = cal_triplet_loss(match(ph_fea, sk_fea), opt.margin)
            loss_tracker.update(loss.item(), ph.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tqdm_batch.close()

        logger.info(
            "Epoch[{}/{}] loss: {:.8f}, avg_loss:{:.8f}.".format(
                i, opt.epoch, loss_tracker.val, loss_tracker.avg
            )
        )

        # test
        acc_1, acc_5, acc_10 = get_retrieval_acc(
            ph_encoder,
            sk_encoder,
            test_ph_loader,
            test_sk_loader,
            i,
            self_interaction,
            match,
        )

        if acc_1 > best_acc[0]:
            best_acc = [acc_1, acc_5, acc_10]
            names = ["ph_encoder", "sk_encoder"]
            models = [ph_encoder, sk_encoder]
            others = {"best_acc": acc_1, "acc_5": acc_5, "acc_10": acc_10}
            save_model(opt, models, names, i, logger, others, best=True)

        logger.info(
            "Epoch[{}/{}] acc@1:{:.4f}, acc@5:{:.4f}, acc@10:{:.4f}, best_acc:{:.4f} {:.4f} {:.4f}".format(
                i,
                opt.epoch,
                acc_1,
                acc_5,
                acc_10,
                best_acc[0],
                best_acc[1],
                best_acc[2],
            )
        )


if __name__ == "__main__":
    train()
