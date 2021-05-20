#!/bin/sh
echo "monitor the way of test about [get]"
result_dir=$1
exp_dir=$2
exp_source=$3
data_set=$4

old_result_num=0
result_num=0
while [ 1 ]; do
    sleep 200s
    result_num=`ls $result_dir\/$data_set\/test | wc -l`
    if [ $result_num == $old_result_num ]
    then
      echo "get mode is dead,the number of patients tested is {$old_result_num}"
      get_mode_pid=`ps -ef | grep python | grep test_patients | awk '{print $2}'`
      echo $get_mode_pid
      `kill -9 $get_mode_pid`
      get_mode_pid=`ps -ef | grep python | grep test_patients | awk '{print $2}'`
      if [ ! $get_mode_pid ]
      then
        echo "killed the dead pids {$get_mode_pid}"
        break
      fi
    else
      old_result_num=$result_num
    fi
done

# query mode
echo "begin to query unnormal case and to test by 1 process"
python  -m test_patients --fun query --data_set $data_set \
        -result_dir  $result_dir \
        -exp_source  $exp_source \
        -exp_dir  $exp_dir
#res mode
echo "begin to get the metrics"
python  -m test_patients --fun res --data_set $data_set \
        -result_dir  $result_dir \
        -exp_source  $exp_source \
        -exp_dir  $exp_dir
echo "monitor script exit 0"