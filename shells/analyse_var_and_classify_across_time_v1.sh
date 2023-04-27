NEW_TIME="all"
# Classifiers' changes
python -m model.analyse --source_type api --source_dataset HC3 --time after0301 --task visualize \
--end_id 1000 --feature_type classify --vis_task visualize_one_feature \
--diff_type time --return_type hc3_group --knowledge_extractor no --pic_type line --times $NEW_TIME

# acc
python -m model.analyse --source_type api --source_dataset HC3 --time after0301 --task visualize \
--end_id 1000 --feature_type classify --vis_task visualize_one_feature \
--diff_type time --return_type hc3_group --knowledge_extractor no --pic_type line \
--feature_group_type avg_acc --times $NEW_TIME

# top 10 and bottom 10 var_score features
python -m model.analyse --source_type api --source_dataset HC3 --time after0301 --task visualize \
 --end_id 1000 --feature_type linguistic --vis_task diff_snapshots_by_var \
 --diff_type time --pic_topk_feat 10 --return_type var_score --times $NEW_TIME


