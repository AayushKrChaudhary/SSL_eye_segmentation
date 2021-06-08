from pprint import pprint
import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--dataset', type=str, default='Semantic_Segmentation_Dataset', help='name of dataset')
    # Optimization: General
    parser.add_argument('--bs', type=int, default = 4)
    parser.add_argument('--bs_U', type=int, default = 4)
    parser.add_argument('--epochs', type=int,help='Number of epochs',default= 250)
    parser.add_argument('--workers', type=int,help='Number of workers',default=2)
    parser.add_argument('--model', help='model name',default='ritnet')
    parser.add_argument('--evalsplit', help='eval spolit',default='val')
    parser.add_argument('--lr', type=float,default= 1e-3,help='Learning rate')
    parser.add_argument('--save', help='save folder name',default='0try')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load', type=str, default=None, help='load checkpoint file name')
    parser.add_argument('--resume',  action='store_true', help='resume train from load chkpoint')
    parser.add_argument('--test', action='store_true', help='test only')
    parser.add_argument('--savemodel',action='store_true',help='checkpoint save the model')
    parser.add_argument('--testrun', action='store_true', help='test run with few dataset')
    parser.add_argument('--expname', type=str, default='1/ssl_augu_10_1', help='extra explanation of the method')
    parser.add_argument('--useGPU', type=str, default=True, help='Set it as False if GPU is unavailable')
    parser.add_argument('--factor', type=str, default=1, help='Select resolution downsampling factor')
    parser.add_argument('--mode', help='mode',default='ssl_augu') #labeled, ssl, ssl_augu
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--labeltype', default='all', type=str, help ='Select the number of labeled images to be used for training based on Data/..txt filename')
    parser.add_argument('--SegLoss', default=False, type=str, help='Set Dice and Surface Loss') ##2,5,10,100
    parser.add_argument('--SSLvalue', default=5, type=int)
    parser.add_argument('--deviceID', default='0', type=str)    
    

    # parse 
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args

if __name__ == '__main__':

    opt = parse_args()
    print('opt[\'dataset\'] is ', opt.dataset)




