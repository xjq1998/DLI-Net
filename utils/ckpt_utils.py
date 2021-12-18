import os
import torch


def resume_model(resume_path, sketch_model, img_model):
    ckpt = torch.load(resume_path)
    sketch_model.load_state_dict(ckpt['sketch'])
    img_model.load_state_dict(ckpt['img'])
    lr = ckpt['lr']
    epoch = ckpt['epoch']
    avg_loss = ckpt['avg_loss']
    val_acc1 = ckpt['val_acc@1']
    val_acc5 = ckpt['val_acc@5']
    val_acc10 = ckpt['val_acc@10']
    best_acc1 = ckpt['best_acc@1']
    print(
        "====> resume from '{}' (epoch:{}, avg_loss:{}, val_acc@1:{}, val_acc@5:{}, val_acc@10:{}"
        .format(resume_path, epoch, avg_loss, val_acc1, val_acc5, val_acc10))
    return lr, epoch, best_acc1


def save_model(opt, models, names, epoch, logger, others=None, best=False):
    save_root = opt.checkpoint_path
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if best:
        ckpt_path = os.path.join(save_root, 'ckpt-best.pth')
    else:
        ckpt_path = os.path.join(save_root, 'ckpt-{}.pth'.format(epoch))

    save_dict = {'epoch': epoch}
    for i in range(len(names)):
        save_dict[names[i]] = models[i].state_dict()
    if others != None:
        save_dict.update(others)
    torch.save(save_dict, ckpt_path)
    logger.info('model saved in {}'.format(ckpt_path))
