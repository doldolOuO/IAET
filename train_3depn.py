import torch, os, random
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataloader import EPNDataLoader
from config_3depn import params
from models.IAET import IAET
from models.utils import fps_subsample
from cuda.ChamferDistance import L2_ChamferDistance


def set_seed(seed=42):
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        # torch.backends.cudnn.deterministic = True


def main():
    # default setting
    cfg = params()
    MODEL = 'IAET'
    FLAG = 'train_3depn'
    CLASS = 'car'
    BATCH_SIZE = int(cfg.batch_size)

    # create ckpt_dir
    ckpt_dir = f'ckpt_3depn'
    if not os.path.exists(os.path.join(ckpt_dir)):
        os.makedirs(os.path.join(ckpt_dir))

    # create models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IAET()
    model = torch.nn.DataParallel(model)
    model.to(device)

    # loss function
    loss_cd = L2_ChamferDistance()

    # dataset loading
    EPNDataset_train = EPNDataLoader('./dataset/3depn_train_list.txt',
                                     data_path=cfg.data_root,
                                     status="train",
                                     category=CLASS)
    train_loader = DataLoader(EPNDataset_train,
                              batch_size=cfg.batch_size,
                              num_workers=cfg.nThreads,
                              shuffle=True,
                              drop_last=False)

    # optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.7)

    # saving hyperparameters
    CONFIG_FILE = f'ckpt_3depn/CONFIG.txt'
    with open(CONFIG_FILE, 'w') as f:
        f.write('MODEL:' + str(MODEL) + '\n')
        f.write('FLAG:' + str(FLAG) + '\n')
        f.write('CLASS:' + str(CLASS) + '\n')
        f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
        f.write('MAX_EPOCH:' + str(int(cfg.n_epochs)) + '\n')
        f.write(str(cfg.__dict__))

    # training
    set_seed()
    for epoch in range(0, cfg.n_epochs):
        model.train()
        n_batches = len(train_loader)
        with tqdm(train_loader) as t:
            for batch_idx, data in enumerate(t):
                image = data[0].to(device)
                partial = data[2].to(device)
                gt = data[1].to(device)
                out = model(partial, image)
                stage2 = fps_subsample(gt.contiguous(), out[2].size(1))
                stage1 = fps_subsample(stage2.contiguous(), out[1].size(1))
                stage0 = fps_subsample(stage1.contiguous(), out[0].size(1))
                loss_stage0 = loss_cd(stage0, out[0])
                loss_stage1 = loss_cd(stage1, out[1])
                loss_stage2 = loss_cd(stage2, out[2])
                loss_stage3 = loss_cd(gt, out[3])
                loss = loss_stage0 + loss_stage1 + loss_stage2 + loss_stage3
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch,
                                                                  cfg.n_epochs,
                                                                  batch_idx + 1,
                                                                  n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [1e3 * loss_stage0.data.cpu(),
                                                                1e3 * loss_stage1.data.cpu(),
                                                                1e3 * loss_stage2.data.cpu(),
                                                                1e3 * loss_stage3.data.cpu()
                                                                ]])
        print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
    torch.save({'model_state_dict': model.state_dict()},
               f'ckpt_3depn/car.pt')

if __name__ == '__main__':
    main()