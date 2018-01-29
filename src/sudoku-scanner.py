"""
Command line tool for using the sudoku-scanner on saved images
"""
import argparse
import sys

from sudoku import Sudoku


def scan(img_path):
    """Scans image and outputs predictions."""
    sudoku = Sudoku(img_path)
    sudoku.process()
    sudoku.predict()
    print(sudoku)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    path = sys.argv[1]
    scan(path)
