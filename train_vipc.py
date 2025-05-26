import torch, os, time, random
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataloader import ViPCDataLoader
from config_vipc import params
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
    CLASS = cfg.cat
    MODEL = 'IAET'
    FLAG = 'train_vipc'
    BATCH_SIZE = int(cfg.batch_size)
    best_loss = 99999
    resume_epoch = 0

    # create log
    TIME_FLAG = time.asctime(time.localtime(time.time()))
    log_dir = f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}'
    if not os.path.exists(os.path.join(log_dir)):
        os.makedirs(os.path.join(log_dir))

    # create models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IAET()
    model = torch.nn.DataParallel(model)
    model.to(device)

    # loss function
    loss_cd = L2_ChamferDistance()
    loss_cd_eval = L2_ChamferDistance()

    # dataset loading
    ViPCDataset_train = ViPCDataLoader('./dataset/vipc_train_list.txt',
                                       data_path=cfg.data_root,
                                       status="train",
                                       category=cfg.cat)
    train_loader = DataLoader(ViPCDataset_train,
                              batch_size=cfg.batch_size,
                              num_workers=cfg.nThreads,
                              shuffle=True,
                              drop_last=True)
    ViPCDataset_test = ViPCDataLoader('./dataset/vipc_test_list.txt',
                                      data_path=cfg.data_root,
                                      status="test",
                                      category=cfg.cat)
    test_loader = DataLoader(ViPCDataset_test,
                             batch_size=cfg.batch_size,
                             num_workers=cfg.nThreads,
                             shuffle=False,
                             drop_last=False)

    # optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.7)

    # saving hyperparameters
    CONFIG_FILE = f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/CONFIG.txt'
    with open(CONFIG_FILE, 'w') as f:
        f.write('RESUME:' + str(cfg.resume) + '\n')
        f.write('FLAG:' + str(FLAG) + '\n')
        f.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
        f.write('MAX_EPOCH:' + str(int(cfg.n_epochs)) + '\n')
        f.write('CLASS:' + str(CLASS) + '\n')
        f.write(str(cfg.__dict__))

    # models loading
    if cfg.resume:
        ckpt_dict = torch.load(cfg.ckpt)
        model.load_state_dict(ckpt_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        resume_epoch = ckpt_dict['epoch'] + 1
        best_loss = ckpt_dict['loss']

    # training
    set_seed()
    for epoch in range(resume_epoch, cfg.n_epochs):
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
        if epoch % int(cfg.eval_epoch) == 0:
            with torch.no_grad():
                model.eval()
                Loss = 0
                with tqdm(test_loader) as t:
                    for batch_idx, data in enumerate(t):
                        image = data[0].to(device)
                        partial = data[2].to(device)
                        gt = data[1].to(device)
                        out = model(partial, image)
                        loss = loss_cd_eval(out[-1], gt)
                        Loss += loss * 1e3
                    Loss = Loss / len(test_loader)
                    if Loss < best_loss:
                        best_loss = Loss
                        best_epoch = epoch
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': Loss},
                            f'./log/{MODEL}_{BATCH_SIZE}_{CLASS}_{FLAG}_{TIME_FLAG}/ckpt_{epoch}_{Loss}.pt')
                        print('best epoch: ', best_epoch, 'cd: ', best_loss.item())
                    print(epoch, ' ', Loss.item(),
                          'lr: ', optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

if __name__ == '__main__':
    main()