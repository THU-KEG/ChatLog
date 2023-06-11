#for语句结构
#NEW_SEED=
for s in 1 2 66 49 18
do
  for r in bottom10_single random10_single single_ppl_gltr
  do
    python -m model.models --source_type api --source_dataset HC3 --task train --end_id 1000 --feature_type all --knowledge_extractor no --diff_type time --return_type dict --diff_type time --train_time_setting only01 --train_human_num 1000 --test_num 1000 --train_feature_setting $r --seed $s
  done
done


# evaluate
python -m data.evaluation --task calculate_detect_std --seeds 1 2 66 49 18