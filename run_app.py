import sys
import os
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    sys.path.insert(0, base_path)

    from Staj.cli import main
    main()