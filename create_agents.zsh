#!/bin/zsh

for i in {1..$1}
do
    echo "$i"
    ./main.py $i $1
done

