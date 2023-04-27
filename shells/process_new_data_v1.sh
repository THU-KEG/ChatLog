NEW_TIME="2023-04-01 2023-04-02 2023-04-03 2023-04-04 2023-04-05 2023-04-06 2023-04-07 2023-04-08 2023-04-09"

# knowledge, linguistic and classify features
CUDA_VISIBLE_DEVICES=2 python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
 --task extract_feature --end_id 1000 --feature_type all --knowledge_extractor all \
--times $NEW_TIME

# rouge
python -m data.evaluation --source_type api --time after0301 --source_dataset HC3 --task evaluate --times $NEW_TIME
