import os, argparse

def main(args):
    with open(root + 'annotations/old_all.txt', 'r') as f:
        lines = f.read().splitlines()
    
    lines = list(map(lambda s: s.split(','), lines))
    for i in range(len(lines)-1):
        if lines[i][0]==lines[i+1][0]:
            print('Duplication found at {}.'.format(i))
    
    print('{} lines of data in total.'.format(len(lines)))


def _parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--index', '-i', type=int, required=True,
    #     help='Index of the requested data.'
    # )
    return parser.parse_args()


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data/sys_ucc/')
    args = _parse_args()
    main(args)