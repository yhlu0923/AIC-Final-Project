#! /bin/sh
if [ $# -ne 1 ] || ! [ -d "./$1" ]; then
    echo "usage: $1 <output directory>"
    exit
fi

annots="./$1/test_annots.csv"
if ! [ -f "$annots" ]; then
    annots="./$1/val_annots.csv"
fi

echo "Predict with $annots."

python3 ./visualize.py \
    --dataset csv \
    --csv_classes ./class_list.csv \
    --csv_val "$annots" \
    --model "./$1/model_final.pt"
