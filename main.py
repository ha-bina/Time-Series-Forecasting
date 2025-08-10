import argparse
from scripts import task1_preprocess_eda


def cli():
    parser = argparse.ArgumentParser(description='Financial Analysis Pipeline')
    parser.add_argument('--task', type=int, default=1, help='Task number to run (1)')
    args = parser.parse_args()

    if args.task == 1:
        task1_preprocess_eda.main()
    else:
        raise NotImplementedError('Only task 1 is implemented in this repo scaffold')


if __name__ == '__main__':
    cli()