"""
Command line tool for using the sudoku-scanner on saved images
"""
import argparse

from sudoku import Sudoku


def scan(img_path):
    """Scans image and outputs predictions."""
    sudoku = Sudoku(img_path)
    sudoku.process()
    sudoku.predict()
    return sudoku


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scan a sudoku and (by default) print to stdout')
    parser.add_argument(
        'file',
        type=argparse.FileType('r'),
        help='Path to an image of a sudoku')
    parser.add_argument(
        '-o', '--output',
        type=argparse.FileType('w'),
        default=None,
        help='Write file instead of printing to stdout')

    args = parser.parse_args()

    sudoku = scan(args.file.name)


    if args.output:
        args.output.write(sudoku.get_predictions())
        #with open(args.output, 'w') as f:
            #f.write(str(sudoku.predictions))
    else:
        print(sudoku)
