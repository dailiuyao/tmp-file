#!/bin/bash

for file in *.pdf; do
	pdfcrop "$file" "${file%.pdf}-crop.pdf"
done
