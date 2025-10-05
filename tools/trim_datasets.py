import os
import csv
import argparse


def trim_csv_to_size(src_path, dst_path, max_mb=100):
    max_bytes = max_mb * 1024 * 1024
    with open(src_path, 'r', encoding='utf-8', errors='replace') as src, open(dst_path, 'w', encoding='utf-8', newline='') as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)

        # write header
        try:
            header = next(reader)
        except StopIteration:
            return 0
        writer.writerow(header)

        written = dst.tell()
        for row in reader:
            writer.writerow(row)
            written = dst.tell()
            if written >= max_bytes:
                break

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--max_mb', type=int, default=100)
    args = parser.parse_args()

    for f in args.files:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            continue
        dirname = os.path.dirname(f)
        base = os.path.basename(f)
        name, ext = os.path.splitext(base)
        out = os.path.join(dirname, f"{name}_trimmed{ext}")
        written = trim_csv_to_size(f, out, max_mb=args.max_mb)
        print(f"Wrote {written} bytes to {out}")


if __name__ == '__main__':
    main()
