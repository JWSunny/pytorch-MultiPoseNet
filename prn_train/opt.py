import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):

        # --------------------------  General Training Options
        self.parser.add_argument('--lr', type=float, default=1.0e-3, help='Learning Rate')
        self.parser.add_argument('--lr_gamma', type=float, default=0.9, help='Gamma Rate')
        self.parser.add_argument('--number_of_epoch', type=int, default=16)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--node_count', type=int, default=1024, help='Hidden Layer Node Count')
        # --------------------------  General Training Options

        self.parser.add_argument('--exp', type=str, default='test/', help='Experiment name')

        # --------------------------
        self.parser.add_argument('--coeff', type=int, default=2, help='Coefficient of bbox size')
        self.parser.add_argument('--threshold', type=int, default=0.21, help='BBOX threshold')
        self.parser.add_argument('--test_cp', type=str,default='checkpoint/test/epoch16checkpoint.pth.tar' ,help='Path to model for testing')
        self.parser.add_argument('--num_of_keypoints', type=int, default=3, help='Minimum number of keypoints for each bbox in training')
        self.parser.add_argument('--test_keypoint_count', type=int, default=0, help='Validating with different keypoint count')
        self.parser.add_argument('--window_size', type=int, default=15, help='Windows size for cropping')
        # --------------------------

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt