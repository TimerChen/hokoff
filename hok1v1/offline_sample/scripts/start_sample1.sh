#!/bin/bash

# export INTERFACE_SO_NOT_USE_CURVE=1

SCRIPT_DIR=$(dirname $(dirname $(readlink -f $0)))

if [ ! -n "$1" ] ### if exists one parameter ###
then
    LEVEL_STR="8,0"
else
    LEVEL_STR=$1
fi
echo "LEVLES:$LEVEL_STR"

if [ ! -n "$2" ] 
then
    EVAL_NUMBER=1
else
    EVAL_NUMBER=$2
fi
echo "EVAL_NUMBER:$EVAL_NUMBER"
if [ ! -n "$3" ] 
then
    CPU_NUMBER=1
else
    CPU_NUMBER=$3
fi
echo "CPU_NUMBER:$CPU_NUMBER"
if [ ! -n "$4" ] 
then
    DATASET_VERSION_NAME='tmpversion1'
else
    DATASET_VERSION_NAME=$4
fi

A0=`echo $LEVEL_STR | awk -F, '{print $1}'`
A1=`echo $LEVEL_STR | awk -F, '{print $2}'`
if [ ! -n "$5" ] 
then
    DATASET_NAME="level-$A0-$A1/"
else
    DATASET_NAME=$5
fi
echo "DATASET_NAME:$DATASET_NAME"
LOG_DIR=$SCRIPT_DIR/logs/$DATASET_NAME
rm -rf $LOG_DIR
mkdir -p $LOG_DIR ### recursive make dirs ###
cd $SCRIPT_DIR/scripts
set -x
bash -x sample1.sh $LEVEL_STR $EVAL_NUMBER $CPU_NUMBER $DATASET_VERSION_NAME $DATASET_NAME
