#for语句结构
for n in top5 bottom5 top5_bottom5 top10 bottom10 top10_bottom10 no_limit
do
  python -m model.models --source_type api --source_dataset HC3 --task train --end_id 1000 --feature_type linguistic --knowledge_extractor no --diff_type time --return_type dict --diff_type time --train_time_setting only01 --train_human_num 1000 --test_num 1000 --train_feature_setting $n
done

for s in 1 2 3
do
  for r in random5 random10
  do
    python -m model.models --source_type api --source_dataset HC3 --task train --end_id 1000 --feature_type linguistic --knowledge_extractor no --diff_type time --return_type dict --diff_type time --train_time_setting only01 --train_human_num 1000 --test_num 1000 --train_feature_setting $r --seed $s
  done
done

