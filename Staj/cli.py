import argparse
import time
import os
import logging
from pathlib import Path
from .config import config
from .pipeline import Pipeline, ProgressInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_cli(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("GÖRÜNTÜ İYİLEŞTİRİCİ - HEADLESS MODE")
    print("=" * 60)

    if args.config:
        config.load(args.config)
    
    print(f"Giriş: {args.input}")
    print(f"Çıkış: {args.output}")
    print(f"Mod: {args.mode}")
    print(f"Config: {args.config or 'default'}")
    print("=" * 60)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline()
    
    def on_progress(info: ProgressInfo) -> None:
        eta_str = f"{info.eta:.0f}sn" if info.eta < 60 else f"{info.eta/60:.1f}dk"
        print(f"\r[{info.current}/{info.total}] {info.filename} | "
              f"Hız: {info.speed:.1f}/sn | ETA: {eta_str}", end="", flush=True)
    
    def on_complete(processed: int, errors: int, error_list: list) -> None:
        print()
        print("=" * 60)
        print(f"TAMAMLANDI")
        print(f"İşlenen: {processed}")
        print(f"Hata: {errors}")
        if error_list:
            print("\nHatalı dosyalar:")
            for filename, msg in error_list[:10]:
                print(f"  - {filename}: {msg}")
            if len(error_list) > 10:
                print(f"  ... ve {len(error_list) - 10} hata daha")
        print("=" * 60)
    
    pipeline.on_progress = on_progress
    pipeline.on_complete = on_complete

    skip = not args.no_skip
    count = pipeline.scan_files(args.input, skip, args.output)
    
    if count == 0:
        print("İşlenecek dosya yok")
        return
    
    print(f"İşlenecek: {count} dosya")

    if not args.no_histogram:
        pipeline.prepare_histogram(args.input, True)

    start = time.time()
    pipeline.run_parallel(args.input, args.output, args.mode, None)
    
    elapsed = time.time() - start
    print(f"Toplam süre: {elapsed:.1f} saniye")
    print(f"Ortalama hız: {pipeline.processed / elapsed:.1f} dosya/sn")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Görüntü İyileştirici",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Guide:
  # GUI ile çalıştır
  python -m Staj
  
  # Headless modunda çalıştır
  python -m Staj --input ./raw --output ./enhanced --headless
  
  # Kişisel config ile
  python -m Staj -i ./raw -o ./enhanced --headless -c my_config.json
  
  # Default config oluştur
  python -m Staj --generate-config
        """
    )
    
    parser.add_argument("--input", "-i", type=str, help="Giriş klasörü")
    parser.add_argument("--output", "-o", type=str, help="Çıkış klasörü")
    parser.add_argument("--mode", "-m", type=str, default="auto",
                       choices=["auto", "manual"], help="İşleme modu")
    parser.add_argument("--config", "-c", type=str, help="Config dosyası")
    parser.add_argument("--headless", action="store_true", help="GUI olmadan çalıştır")
    parser.add_argument("--no-skip", action="store_true", help="Var olan dosyaları üzerine yaz")
    parser.add_argument("--no-histogram", action="store_true", help="Histogram eşitlemeyi kapat")
    parser.add_argument("--generate-config", action="store_true", help="Default config oluştur")
    
    args = parser.parse_args()
    
    if args.generate_config:
        config.save("config.json")
        print("config.json oluşturuldu")
        return
    
    if args.headless:
        if not args.input or not args.output:
            parser.error("--headless için --input ve --output gerekli")
        run_cli(args)
    else:
        from .gui import run_gui
        run_gui()


if __name__ == "__main__":
    main()
