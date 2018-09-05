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
        # --------------------------  General Training Options

        self.parser.add_argument('--exp', type=str, default='test/', help='Experiment name')

        # --------------------------
        self.parser.add_argument('--cocopath',type=str,default='E:/datasets/2017coco/train2017/')
        self.parser.add_argument('--num_of_keypoints', type=int, default=3, help='Minimum number of keypoints for each bbox in training')
        self.parser.add_argument('--test_keypoint_count', type=int, default=0, help='Validating with different keypoint count')
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