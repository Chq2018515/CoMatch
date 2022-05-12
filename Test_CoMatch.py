import argparse
from numpy import append, arange
import torch
from WideResNet import WideResnet
from datasets.custom import get_test_loader
from utils import AverageMeter, get_fp, get_recall


def my_evaluate(model, dataloader, flag, thresh = 0.9):
    model.eval()
    res = AverageMeter()
    measure = get_fp if flag == "fp" else get_recall
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            
            logits, _ = model(ims)
            # 1 softmax
            # scores = torch.softmax(logits, dim=1)

            # 2 minmax
            upper = torch.max(logits,dim=1).values.view(-1,1).expand_as(logits)
            lower = torch.min(logits,dim=1).values.view(-1,1).expand_as(logits)
            scores = (logits - lower) / (upper - lower) #  0.95      70.83882007519357       71.55987394957984
            
            # 3 gaussion
            # mean = torch.mean(logits)
            # std = torch.std(logits)
            # scores = (logits - mean) / std
            scores = scores / torch.sum(scores, dim=1).view(-1,1).expand_as(scores)
            
            res.update(measure(scores, thresh))
            
    return res.avg


def main():
    parser = argparse.ArgumentParser(description='CoMatch Custom Testing')
    parser.add_argument('--root', default='/opt/chenhaoqing/data/redtheme/batched_data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')    
    parser.add_argument('--dataset', type=str, default='redtheme',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=3,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=512,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)
    
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    
    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')    
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')   
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')   
    parser.add_argument('--alpha', type=float, default=0.9)   
    parser.add_argument('--queue-batch', type=float, default=5, 
                        help='number of batches stored in memory bank')    
    parser.add_argument('--exp-dir', default='CoMatch', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    parser.add_argument('--folds', default='redtheme', type=str, help='guess means k-fold')
    parser.add_argument('--saved-model', default='redtheme/1/CoMatch/checkpoint_80.pth', type=str, help='use saved model')
    
    args = parser.parse_args()
    model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=True)
    model.cuda()  
    parameter = torch.load(args.saved_model)
    model.load_state_dict(parameter, strict=False)
    
    thresh_set = append(arange(0.05,1,0.05),arange(0.955,1,0.005))
    dlfp = get_test_loader(dataset=args.dataset, batch_size=64, num_workers=2, type="fp", root=args.root)
    for thr in thresh_set:
        fp = my_evaluate(model, dlfp, "fp", thr)
        print(thr, "\t", fp)
    dlrecall = get_test_loader(dataset=args.dataset, batch_size=64, num_workers=2, type="recall", root=args.root)
    for thr in thresh_set:
        recall = my_evaluate(model, dlrecall, "recall", thr)
        print(thr, "\t", recall)
if __name__ == '__main__':
    main()