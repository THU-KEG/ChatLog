#for t in tcc pearson
for t in pearson
do
#  for n in classify linguistic
  for n in all
  do
    python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
      --task visualize --end_id 1000 --feature_type $n --vis_task visualize_one_feature \
      --diff_type time --return_type feature_and_eval --knowledge_extractor no --pic_type corr --corr_type $t
  done
done
