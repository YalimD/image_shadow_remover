
from shadow_remover import ShadowRemover

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove shadows from given image",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )
    parser.add_argument('-i', '--image', required=True, help="Image of interest")
    parser.add_argument('-s', '--save', help= "Save the result",
                        default=True)
    parser.add_argument('-v', '--verbose', help="Verbose",
                        default=False)
    parser.add_argument('--rk', help="Region Adjustment Kernel Size", default=10)
    parser.add_argument('--sdk', help="Shadow Dilation Kernel Size", default=3)
    parser.add_argument('--sdi', help="Shadow Dilation Iteration", default=5)
    parser.add_argument('--lab', help="Adjust the pixel values according to LAB",
                        default=False)
    args = parser.parse_args()

    ShadowRemover.process_image_file(*vars(args).values())
