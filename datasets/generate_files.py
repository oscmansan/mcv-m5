import argparse
import os


def segmentation(args):
    # Train
    with open(os.path.join(args.output, 'train_images.txt'), 'w') as f:
        for mask_name in os.listdir(os.path.join(args.input, 'train', 'images')):
            f.write(mask_name + '\n')

    with open(os.path.join(args.output, 'train_labels.txt'), 'w') as f:
        for mask_name in os.listdir(os.path.join(args.input, 'train', 'masks')):
            f.write(mask_name + '\n')

    # Validation
    with open(os.path.join(args.output, 'val_images.txt'), 'w') as f:
        for mask_name in os.listdir(os.path.join(args.input, 'valid', 'images')):
            f.write(mask_name + '\n')

    with open(os.path.join(args.output, 'val_labels.txt'), 'w') as f:
        for mask_name in os.listdir(os.path.join(args.input, 'valid', 'masks')):
            f.write(mask_name + '\n')

    # Test
    if os.path.exists(os.path.join(args.input, 'test')):
        with open(os.path.join(args.output, 'test_images.txt'), 'w') as f:
            for mask_name in os.listdir(os.path.join(args.input, 'test', 'images')):
                f.write(mask_name + '\n')

        with open(os.path.join(args.output, 'test_labels.txt'), 'w') as f:
            for mask_name in os.listdir(os.path.join(args.input, 'test', 'masks')):
                f.write(mask_name + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['segmentation'])
    parser.add_argument('input')
    parser.add_argument('output')

    args = parser.parse_args()

    os.makedirs(args.output)

    if args.mode == 'segmentation':
        segmentation(args)


if __name__ == '__main__':
    main()
