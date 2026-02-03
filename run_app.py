import sys
import os
from multiprocessing import freeze_support

if __name__ == '__main__':
    # Windows EXE için zorunlu kilit
    freeze_support()
    
    # PyInstaller geçici klasör yolu ayarı
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Modül yolunu ekle
    sys.path.insert(0, base_path)

    # Uygulamayı başlat
    from Staj.cli import main
    main()