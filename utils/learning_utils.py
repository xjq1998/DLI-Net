import torch
from tqdm import tqdm

# from einops import repeat, rearrange


class AverageMeter(object):
    """Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_second(secs):
    h = int(secs / 3600)
    m = int((secs % 3600) / 60)
    s = int(secs % 60)
    ss = "(h:m:s):{:0>2}:{:0>2}:{:0>2}".format(h, m, s)
    return ss


def adjust_lr(epoch, optimizer):
    if epoch % 30 == 0:
        optimizer.state_dict()["param_groups"][0]["lr"] *= 0.5


def chair_label_join(label):
    names = label.split("_")
    match_label = ""
    for name in names[:-1]:
        match_label += name
        match_label += "_"
    return match_label[:-1]


def get_retrieval_acc(
    ph_encoder,
    sk_encoder,
    ph_loader,
    sk_loader,
    epoch,
    self_interaction=None,
    match=None,
    dataset="qmul",
):
    ph_encoder.eval()
    sk_encoder.eval()

    with torch.no_grad():
        # processing all test ph
        tqdm_batch = tqdm(ph_loader, desc="epoch-{}-photo-processing".format(epoch))
        ph_list = []
        ph_label_list = []
        for ph, label in tqdm_batch:
            ph = ph.cuda()
            if dataset == "sketchy":
                ph_fea,_ = ph_encoder(ph)
            else:
                ph_fea = ph_encoder(ph)
            # [B,F]
            if self_interaction:
                ph_fea = self_interaction(ph_fea)
            ph_fea = ph_fea.cpu()

            for i in range(ph.shape[0]):
                ph_list.append(ph_fea[i].unsqueeze(0))
                ph_label_list.append(label[i])
        tqdm_batch.close()
        ph_feas = torch.cat(ph_list, dim=0)
        # [ph_num,F]

        # retrival
        sk_num = len(sk_loader.dataset)
        ph_num = len(ph_loader.dataset)
        dist_mat = torch.zeros(sk_num, ph_num)

        correct_num_1 = 0
        correct_num_5 = 0
        correct_num_10 = 0
        tqdm_batch = tqdm(sk_loader, desc="epoch{}-sketch-retival".format(epoch))
        sk_label_list = []
        for i_sk, (sketch, label) in enumerate(tqdm_batch):
            sketch = sketch.cuda()
            if dataset == "sketchy":
                sk_fea,_ = sk_encoder(sketch)
            else:
                sk_fea = sk_encoder(sketch)
            # [B,F]
            if self_interaction:
                sk_fea = self_interaction(sk_fea)
            i_sk = i_sk * sk_loader.batch_size
            j_sk = i_sk + sk_fea.shape[0]
            for i_ph in range(0, ph_num, 32):
                j_ph = min(i_ph + 32, ph_num)
                dist_mat[i_sk:j_sk, i_ph:j_ph] = match(
                    ph_feas[i_ph:j_ph].cuda(), sk_fea
                ).cpu()
            if dataset == "sketchy":
                for i in range(sketch.shape[0]):
                    sk_label = label[i].split('-')
                    if len(sk_label)>2:
                        sk_label_list.append(label[i][:-len(sk_label[-1])-1])
                    else:
                        sk_label_list.append(sk_label[0])
            else:
                for i in range(sketch.shape[0]):
                    if len(label[i].split("_")) > 2:
                        sk_label_list.append(chair_label_join(label[i]))
                    else:
                        sk_label_list.append(label[i].split('_')[0])

        tqdm_batch.close()

        indics = torch.sort(dist_mat)[1]
        # [sk_num,ph_num]

        for i in range(sk_num):
            match_label = []
            for l in indics[i]:
                match_label.append(ph_label_list[l])
                if len(match_label) == 1:
                    correct_num_1 += sk_label_list[i] in match_label
                elif len(match_label) == 5:
                    correct_num_5 += sk_label_list[i] in match_label
                elif len(match_label) == 10:
                    correct_num_10 += sk_label_list[i] in match_label
                    break

        acc_1 = correct_num_1 / sk_num
        acc_5 = correct_num_5 / sk_num
        acc_10 = correct_num_10 / sk_num

    return acc_1, acc_5, acc_10
