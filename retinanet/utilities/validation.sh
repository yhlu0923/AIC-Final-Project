#! /bin/sh
if [ $# -ne 1 ] || ! [ -d "./$1" ]; then
    echo "usage: $0 <output directory>"
    exit
fi

annots="./$1/test_annots.csv"
if ! [ -f "$annots" ]; then
    annots="./$1/val_annots.csv"
fi

python3 csv_validation.py \
    --images_path ./face-mask-detection/images \
    --class_list_path ./class_list.csv \
    --csv_annotations_path "$annots" \
    --model_path "./$1/model_final.pt"
