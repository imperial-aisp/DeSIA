python_script="../main.py"
note="aia_model"

test_size=450

BLOCKS=(ppmf_060190079011001 ppmf_060411220001000 ppmf_450790104081009 ppmf_340057048024043 ppmf_370779707041005 ppmf_060290046042015 ppmf_060730100141012 ppmf_060730187001717 ppmf_130630406161000 ppmf_360810797012000)

for seed in 0 1 2 3 4
do
        for num_aggregates in 113
        do
                for model in mlp svm rf
                do
                        python $python_script --dataset ppmf --filenames ${BLOCKS[@]} --seed $seed --num-aggregates $num_aggregates --test-size $test_size --chosen-gpus -1 --multi-cpus 5 --note $note \
                                desia --stochastic-method stochastic-method --aia-model $model
                done
        done
done