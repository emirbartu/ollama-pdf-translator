#!/usr/bin/env python3
"""
Compress an existing PDF file using PyMuPDF optimization.
This is useful for PDFs that have become bloated after text replacement.
"""

import argparse
import sys
from pathlib import Path
import fitz  # PyMuPDF


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in MB."""
    return filepath.stat().st_size / (1024 * 1024)


def compress_pdf(input_pdf: Path, output_pdf: Path = None, overwrite: bool = False) -> None:
    """Compress a PDF file.

    Args:
        input_pdf: Path to input PDF file
        output_pdf: Path to output PDF file (default: input_compressed.pdf)
        overwrite: Whether to overwrite the input file
    """
    input_pdf = Path(input_pdf)

    if not input_pdf.exists():
        print(f"Error: PDF file not found: {input_pdf}")
        sys.exit(1)

    # Determine output path
    if overwrite:
        output_pdf = input_pdf.with_suffix('.tmp.pdf')
        will_replace = True
    elif output_pdf is None:
        output_pdf = input_pdf.with_stem(f"{input_pdf.stem}_compressed")
        will_replace = False
    else:
        output_pdf = Path(output_pdf)
        will_replace = False

    # Get original size
    original_size = get_file_size_mb(input_pdf)
    print(f"Original file: {input_pdf}")
    print(f"Original size: {original_size:.2f} MB")
    print(f"Compressing...")

    try:
        # Open and save with compression
        doc = fitz.open(input_pdf)
        doc.save(
            output_pdf,
            garbage=4,      # Maximum garbage collection
            deflate=True,   # Compress content streams
            clean=True      # Clean and sanitize PDF structure
        )
        doc.close()

        # Get compressed size
        compressed_size = get_file_size_mb(output_pdf)
        reduction = ((original_size - compressed_size) / original_size) * 100

        print(f"\nCompressed size: {compressed_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        print(f"Saved: {original_size - compressed_size:.2f} MB")

        # Replace original if requested
        if will_replace:
            input_pdf.unlink()
            output_pdf.rename(input_pdf)
            print(f"\nOriginal file replaced: {input_pdf}")
        else:
            print(f"\nCompressed file saved as: {output_pdf}")

    except Exception as e:
        print(f"Error compressing PDF: {e}")
        # Clean up temp file if it exists
        if will_replace and output_pdf.exists():
            output_pdf.unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compress PDF files using maximum compression settings"
    )
    parser.add_argument(
        "input_pdf",
        help="Path to input PDF file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output PDF file (default: input_compressed.pdf)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the input file with compressed version"
    )

    args = parser.parse_args()

    compress_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
