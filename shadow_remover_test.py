
from shadow_remover import process_image_file

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove shadows from given image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image', required=True, help="Image of interest")
    parser.add_argument('-s', '--save', help= "Save the result",
                        default=True, nargs='?')
    parser.add_argument('--ab', help="AB channel mean sum threshold", type=int, default=256)
    parser.add_argument('--lab', help="Adjust the pixel values according to LAB",
                        default=False, nargs='?')
    parser.add_argument('--rk', help="Region Adjustment Kernel Size", type=int, default=10)
    parser.add_argument('--sdk', help="Shadow Dilation Kernel Size", type=int, default=3)
    parser.add_argument('--sdi', help="Shadow Dilation Iteration", type=int, default=5)
    parser.add_argument('--sst', help="Shadow size threshold", type=int, default=2500)
    parser.add_argument('-v', '--verbose', help="Verbose", const=True,
                        default=False, nargs='?')
    args = parser.parse_args()

    process_image_file(*vars(args).values())
