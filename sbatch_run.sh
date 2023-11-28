for name in rte mrpc
do
    export NAME=$name
    sbatch myjob.sh
done
