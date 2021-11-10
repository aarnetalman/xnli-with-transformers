for model in bert xlmroberta
do
    sbatch train.sh $model
done
