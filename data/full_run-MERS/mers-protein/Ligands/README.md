Converted all files with

for file in *; do obabel $file -O "${file%.*}.sdf"; done
