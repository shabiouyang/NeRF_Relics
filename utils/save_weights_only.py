import argparse

import paddle


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='checkpoint path')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    checkpoint = paddle.load(args.ckpt_path, map_location='cpu')
    paddle.save(checkpoint['state_dict'], args.ckpt_path.split('/')[-2] + '.ckpt')
    print('Done!')
