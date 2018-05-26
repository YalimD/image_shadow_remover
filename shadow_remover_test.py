
from shadow_remover.ShadowRemoval import *

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove shadows from given image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('-i', '--image', help="Image of interest")
    parser.add_argument('-v', '--verbose', help="Verbose", const= True,
                        default=False, nargs='?')
    parser.add_argument('--rk', help="Region Adjustment Kernel Size", default=10)
    parser.add_argument('--sdk', help="Shadow Dilation Kernel Size", default=3)
    parser.add_argument('--sdi', help="Shadow Dilation Iteration", default=5)
    parser.add_argument('--lab', help="Adjust the pixel values according to LAB", const= True,
                        default=False, nargs='?')
    args = parser.parse_args()

    if args.image is None:
        print("Usage -i imgPath")
    else:
        ShadowRemover.removeShadows(*vars(args).values())
