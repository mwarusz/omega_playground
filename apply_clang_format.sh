#!/bin/bash
script_dir=`dirname "$0"`
for file in `find $script_dir \( -name \*.hpp -or -name \*.cpp \) -not -path "*/build/*"`; do 
	clang-format --style=llvm -i $file
done
