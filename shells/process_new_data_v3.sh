# NEW_TIME="2023-04-17 2023-04-18 2023-04-19 2023-04-20 2023-04-21 2023-04-22 2023-04-23"
# NEW_TIME="2023-05-01 2023-05-02 2023-05-03 2023-05-04 2023-05-05 2023-05-09 2023-05-07 2023-05-08"
NEW_TIME="2023-09-09 2023-09-10"
SUFFIX="base1 base2 base3"

# knowledge, linguistic and classify features
CUDA_VISIBLE_DEVICES=7 python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
 --task extract_feature --end_id 1000 --feature_type all --knowledge_extractor all --pp_suffix $SUFFIX \
--times $NEW_TIME

# rouge
python -m data.evaluation --source_type api --time after0301 --source_dataset HC3 --task evaluate --times $NEW_TIME --pp_suffix $SUFFIX
