#!/bin/bash

# LaTeX Compilation Script for Academic Paper
# This script compiles the LaTeX document with proper bibliography handling

echo "Compiling LaTeX Academic Paper..."
echo "====================================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex not found. Please install a LaTeX distribution (e.g., texlive-full)"
    echo "On Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "On macOS: brew install mactex"
    exit 1
fi

# Navigate to the assignment directory
cd "$(dirname "$0")"

# File name (without extension)
FILENAME="Academic_Paper_Report"

echo "Compiling $FILENAME.tex..."

# First pass: Generate auxiliary files
echo "First pass: Generating auxiliary files..."
pdflatex -interaction=nonstopmode "$FILENAME.tex" > compile.log 2>&1

if [ $? -ne 0 ]; then
    echo "First pass failed. Check compile.log for errors."
    tail -20 compile.log
    exit 1
fi

# Second pass: Process bibliography (if bibtex file exists)
if [ -f "$FILENAME.bib" ]; then
    echo "Processing bibliography..."
    bibtex "$FILENAME" >> compile.log 2>&1
fi

# Third pass: Update references
echo "Second pass: Updating references..."
pdflatex -interaction=nonstopmode "$FILENAME.tex" >> compile.log 2>&1

if [ $? -ne 0 ]; then
    echo "Second pass failed. Check compile.log for errors."
    tail -20 compile.log
    exit 1
fi

# Fourth pass: Final compilation
echo "Final pass: Generating final PDF..."
pdflatex -interaction=nonstopmode "$FILENAME.tex" >> compile.log 2>&1

if [ $? -ne 0 ]; then
    echo "Final pass failed. Check compile.log for errors."
    tail -20 compile.log
    exit 1
fi

# Check if PDF was generated successfully
if [ -f "$FILENAME.pdf" ]; then
    echo "Success! PDF generated: $FILENAME.pdf"
    echo ""
    echo "Document Statistics:"
    echo "File size: $(du -h "$FILENAME.pdf" | cut -f1)"
    echo "Pages: $(pdfinfo "$FILENAME.pdf" 2>/dev/null | grep Pages | awk '{print $2}' || echo "Unknown")"
    echo ""
    echo "Generated files:"
    ls -la "$FILENAME".* | grep -E "\.(pdf|log|aux|bbl|blg)$"
    echo ""
    echo "Compilation completed successfully!"
    echo "You can now open $FILENAME.pdf to view your academic paper."
else
    echo "Error: PDF file was not generated."
    echo "Check compile.log for detailed error information."
    exit 1
fi

# Clean up auxiliary files (optional)
read -p "Clean up auxiliary files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up auxiliary files..."
    rm -f "$FILENAME.aux" "$FILENAME.log" "$FILENAME.bbl" "$FILENAME.blg" "$FILENAME.out" "$FILENAME.toc" compile.log
    echo "Cleanup completed."
fi

echo ""
echo "Your academic paper is ready!"
echo "Location: $(pwd)/$FILENAME.pdf"