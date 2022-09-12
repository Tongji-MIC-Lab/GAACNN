import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import Model.model as model
import Util.torch_msssim as torch_msssim
from Model.context_model import Weighted_Gaussian
import time
from dataset import ImageFolder, TestKodakDataset
# from torchvision.datasets import ImageFolder
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device_ids = [0,1,2,3]

def psnr(img1, img2):
    mse = np.mean( np.square(img1 - img2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 10 * math.log10(PIXEL_MAX**2 / mse)


def adjust_learning_rate_flicker(optimizer,step_count):
    if step_count<250000:
        lr=1e-4
    elif step_count < 350000:
        lr = 5e-5
    elif step_count < 450000:
        lr = 2.5e-5
    elif step_count < 550000:
        lr = 1e-5
    elif step_count < 650000:
        lr = 5e-6
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# def adjust_learning_rate(optimizer, epoch, init_lr):
#     """Sets the learning rate to the initial LR decayed by 2 every 3 epochs"""
#     if epoch < 10:
#         lr = init_lr
#     else:
#         lr = init_lr * (0.5 ** ((epoch-7) // 3))
#     if lr < 1e-6:
#         lr = 1e-6
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


def train(args):

    train_data = ImageFolder(root='/root/dataset/flicker_2W_images_crop256', transform=transforms.Compose(
        [transforms.ToTensor()]))        #
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8)

    image_comp = model.Image_coding(3, args.M, args.N2).cuda() # ,nf,nb,gc=32
    context = Weighted_Gaussian(args.M).cuda()

    model_existed = os.path.exists(os.path.join(args.out_dir, 'mse' + str(int(args.lmbda* 100)) + r'.pkl')) and \
                    os.path.exists(os.path.join(args.out_dir, 'mse' + str(int(args.lmbda * 100)) + r'p.pkl'))

    if model_existed:
        image_comp.load_state_dict(torch.load(os.path.join(args.out_dir, 'mse' + str(int(args.lmbda  * 100)) + r'.pkl')))
        context.load_state_dict(torch.load(os.path.join(args.out_dir, 'mse' + str(int(args.lmbda  * 100)) + r'p.pkl')))
        print('resumed the previous model')

    image_comp = nn.DataParallel(image_comp, device_ids=device_ids)
    context = nn.DataParallel(context, device_ids=device_ids)

    opt1 = torch.optim.Adam(image_comp.parameters(), lr=args.lr_coding)
    opt2 = torch.optim.Adam(context.parameters(), lr=args.lr_entropy)

    lamb = args.lmbda
    msssim_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    loss_func = nn.MSELoss().cuda()
    # writer=SummaryWriter(log_dir='./training_gcn')
    steps_c = args.steps

    for epoch in range(15):
        rec_loss_tmp = 0
        last_time = time.time()
        train_bpp3_tmp = 0
        train_bpp2_tmp = 0
        mse_tmp1 = 0
        msssim_tmp1 = 0
        # cur_lr = adjust_learning_rate(opt1, epoch,args.lr)
        # _ = adjust_learning_rate(opt2, epoch,args.lr)
        for step, batch_x in enumerate(train_loader):
            # batch_x = batch_x[0]
            num_pixels = batch_x.size()[0] *batch_x.size()[2] * batch_x.size()[3]
            batch_x = Variable(batch_x).cuda()
            step_count = epoch * len(train_loader) + step + steps_c
            cur_lr=adjust_learning_rate_flicker(opt1, step_count)
            _ = adjust_learning_rate_flicker(opt2, step_count)
            start_time=time.time()

            fake, xp1, xp2, xq1, x3= image_comp(batch_x, 1)

            xp3, _ = context(xq1, x3)


            # MS-SSIM
            # dloss = 1.0 - loss_func(fake, batch_x)
            dloss1 = loss_func(fake, batch_x)
            msssim1 = msssim_func(fake, batch_x)

            train_bpp1 = torch.sum(torch.log(xp1)) / (-np.log(2) * num_pixels)
            train_bpp2 = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels)
            train_bpp3 = torch.sum(torch.log(xp3)) / (-np.log(2) * num_pixels)
            l_rec = 0.1 * train_bpp3 + 0.1 * train_bpp2 + lamb *dloss1

            opt1.zero_grad()
            opt2.zero_grad()
            l_rec.backward()


            # gradient clip
            torch.nn.utils.clip_grad_norm_(image_comp.parameters(), 2)
            torch.nn.utils.clip_grad_norm_(context.parameters(), 2)

            opt1.step()
            opt2.step()

            rec_loss_tmp += l_rec.item()
            mse_tmp1 += dloss1.item()
            msssim_tmp1 += msssim1.item()
            train_bpp3_tmp += train_bpp3.item()
            train_bpp2_tmp += train_bpp2.item()
            # if step%100 ==0:
            #     writer.add_scalar('loss:',rec_loss_tmp/(step+1),epoch*len(train_loader)+step)
            #     writer.add_scalar('bpp_total',(train_bpp3_tmp + train_bpp2_tmp) / (step + 1),epoch*len(train_loader)+step)
            #     writer.add_scalar('psnr:',10.0 * np.log10(1. / (mse_tmp1 / (step + 1))),epoch*len(train_loader)+step)
            #     writer.add_scalar('ms_ssim:',-10 * np.log10(1 - (msssim_tmp1 / (step + 1))),epoch*len(train_loader)+step)

            if step % 100 == 0:
                with open(os.path.join(args.out_dir, 'train_mse' + str(int(args.lmbda * 100)) + '.log'), 'a') as fd:
                    time_used = time.time() - last_time
                    last_time = time.time()
                    mse1 = mse_tmp1 / (step + 1)
                    psnr1 = 10.0 * np.log10(1. / mse1)
                    msssim_dB1 = -10 * np.log10(1 - (msssim_tmp1 / (step + 1)))
                    bpp_total = (train_bpp3_tmp + train_bpp2_tmp) / (step + 1)
                    fd.write(
                        'ep:%d step:%d time:%.1f cur_lr:%.8f loss:%.6f MSE:%.6f bpp_main:%.4f bpp_hyper:%.4f bpp_total:%.4f psnr:%.2f msssim:%.2f\n'
                        % (epoch, step_count, time_used, cur_lr,rec_loss_tmp / (step + 1), mse1, train_bpp3_tmp / (step + 1),
                           train_bpp2_tmp / (step + 1), bpp_total, psnr1,msssim_dB1))
                    # print('ep:%d step:%d time:%.1f lr:%.8f loss:%.6f MSE:%.6f bpp_main:%.4f bpp_hyper:%.4f bpp_total:%.4f psnr:%.2f msssim:%.2f'
                    #       % (epoch, step_count, time_used, cur_lr, rec_loss_tmp / (step + 1), mse1, train_bpp3_tmp / (step + 1),
                    #        train_bpp2_tmp / (step + 1), bpp_total, psnr1, msssim_dB1))
                fd.close()
            # print('epoch', epoch, 'step:', step, 'MSE:', dloss.item(), 'entropy1_loss:', train_bpp1.item(),
            #      'entropy2_loss:', train_bpp2.item(), 'entropy3_loss:', train_bpp3.item())
            if (step + 1) % 2000 == 0:
                torch.save(context.module.state_dict(),
                           os.path.join(args.out_dir, 'mse' + str(int(args.lmbda * 100)) + r'p.pkl'))
                torch.save(image_comp.module.state_dict(),
                           os.path.join(args.out_dir, 'mse' + str(int(args.lmbda * 100)) + r'.pkl'))

            if step_count%5000==0:
                print("the evaluation of %s/5000 step\n" %step_count)
                test_dataset = TestKodakDataset(data_dir=args.val_path)
                test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True)

                with torch.no_grad():
                    image_comp.eval()
                    context.eval()

                    sumBpp = 0
                    sumPsnr = 0
                    sumMsssim = 0
                    sumMsssimDB = 0
                    cnt = 0
                    for batch_idx, input in enumerate(test_loader):
                        num_pixels = input.size()[0] * input.size()[2] * input.size()[3]
                        input = Variable(input).cuda()
                        fake_eval, xp1_eval, xp2_eval, xq1_eval, x3_eval = image_comp(input, 2)
                        xp3_eval, _ = context(xq1_eval, x3_eval)

                        bpp_hyper = torch.sum(torch.log(xp2_eval)) / (-np.log(2) * num_pixels)
                        bpp_main = torch.sum(torch.log(xp3_eval)) / (-np.log(2) * num_pixels)
                        bpp = bpp_hyper + bpp_main

                        psnr_value = psnr(fake_eval.cpu().numpy() * 255.0, input.cpu().numpy() * 255.0)
                        sumBpp += bpp
                        sumPsnr += psnr_value
                        msssim_func = torch_msssim.MS_SSIM(max_val=1.0).cuda()

                        msssim = msssim_func(fake_eval, input)
                        msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                        sumMsssimDB += msssimDB
                        sumMsssim += msssim
                        cnt += 1

                    sumBpp /= cnt
                    sumPsnr /= cnt
                    sumMsssim /= cnt
                    sumMsssimDB /= cnt
                    print("Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}\n".format(
                            cnt, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
                image_comp.train()
                context.train()
    # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=192, help="the value of M")
    parser.add_argument("--N2", type=int, default=128, help="the value of N")
    parser.add_argument("--lambda", type=float, default=24, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--lr_coding", type=float, default=1e-4, help="initial learning rate.")
    parser.add_argument("--lr_entropy", type=float, default=1e-4, help="initial learning rate.")
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--steps', type=int,default=0,help='the number of steps in training process')
    parser.add_argument('--val', dest='val_path', required=True, help='the path of validation dataset')

    args = parser.parse_args()
    print(args)
    train(args)
