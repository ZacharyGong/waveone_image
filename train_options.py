import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', '-df', required=True, type=str,
                    help='data folder')

parser.add_argument('--batch_size', type=int, default=2, 
                    help='Batch size.')

parser.add_argument('--shuffle', type=bool, default=False, 
                    help='shuffle.')

parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Learning rate.')

parser.add_argument('--checkpoint', type=str, default="None",
                    help='full path to the last checkpoint')
		    
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from epoch No.X.')
		    
parser.add_argument('--num_epochs', type=int, default=1,
                    help='num of epochs.')

parser.add_argument('--exp_name', type=str, default="default",
                    help='exp name.')

parser.add_argument('--model', type=str, required=True,
                    help='model name.')

parser.add_argument('--gpus', default='0', type=str,
                    help='GPU indices separated by comma, e.g. \"0,1\".')
		    
parser.add_argument('--parallel', default=False, type=bool,
                    help='data parallel state.')