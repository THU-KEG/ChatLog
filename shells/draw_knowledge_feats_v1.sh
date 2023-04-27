#KC="2023-04-01 2023-04-02 2023-04-03 2023-04-04 2023-04-05 2023-04-06 2023-04-07 2023-04-08 2023-04-09"
NEW_TIME="all"
#NEW_TIME="2023-04-01 2023-04-02 2023-04-03 2023-04-04 2023-04-05 2023-04-06 2023-04-07 2023-04-08 2023-04-09"
for n in $NEW_TIME
do
  python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
    --task visualize --end_id 1000 --feature_type knowledge --vis_task visualize_one_feature \
    --times $n \
    --diff_type time --return_type feature_and_eval --knowledge_extractor all --pic_type corr
done

