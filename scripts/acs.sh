python_script="../main.py"
note="acs"

test_size=450

BLOCKS=(acs_2702 acs_3103 acs_5100)

for seed in 0 1 2 3 4
do
        for num_aggregates in 113 
        do
                python $python_script --dataset acs --filenames ${BLOCKS[@]} --seed $seed --num-aggregates $num_aggregates --test-size $test_size --chosen-gpus -1 --multi-cpus 5 --note $note \
                        cip
                python $python_script --dataset acs --filenames ${BLOCKS[@]} --seed $seed --num-aggregates $num_aggregates --test-size $test_size --chosen-gpus -1 --multi-cpus 5 --note $note \
                        cip --init-solver

                python $python_script --dataset acs --filenames ${BLOCKS[@]} --seed $seed --num-aggregates $num_aggregates --test-size $test_size --chosen-gpus 0 --multi-cpus 5 --note $note \
                        rap
                python $python_script --dataset acs --filenames ${BLOCKS[@]} --seed $seed --num-aggregates $num_aggregates --test-size $test_size --chosen-gpus 0 --multi-cpus 5 --note $note \
                        rap --warm-start

                python $python_script --dataset acs --filenames ${BLOCKS[@]} --seed $seed --num-aggregates $num_aggregates --test-size $test_size --chosen-gpus -1 --multi-cpus 5 --note $note \
                        desia --stochastic-method stochastic-method
        done
done