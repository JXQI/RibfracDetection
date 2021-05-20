exp_dir="experiments/rifrac_RetinaNet_debug"
exp_source="experiments/rifrac_RetinaNet_exp"
result_dir="/media/victoria/9c3e912e-22e1-476a-ad55-181dbde9d785/jinxiaoqiang/rifrac/experiment/RetinaNet_3D"
data_set='val'

python  -m test_patients --fun get --data_set $data_set \
        -result_dir  $result_dir \
        -exp_source  $exp_source \
        -exp_dir  $exp_dir >>run.logs & sh script/RetinaNet_3D_test_patient_unnormal.sh $result_dir $exp_dir $exp_source $data_set >>montior.log



