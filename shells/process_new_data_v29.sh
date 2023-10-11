# NEW_TIME="2023-04-17 2023-04-18 2023-04-19 2023-04-20 2023-04-21 2023-04-22 2023-04-23"
# NEW_TIME="2023-09-11 2023-09-12 2023-09-13 2023-09-14 2023-09-15 2023-09-16 2023-09-17 2023-09-18 2023-09-19 2023-09-20"
# NEW_TIME="2023-09-21 2023-09-22 2023-09-23 2023-09-24 2023-09-25 2023-09-26"
# 2023-09-18 2023-09-19 2023-09-20
NEW_TIME="2023-09-16 2023-09-17"
SUFFIX="base1 base2 base3"

# knowledge, linguistic and classify features
CUDA_VISIBLE_DEVICES=4 python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
 --task extract_feature --end_id 1000 --feature_type all --knowledge_extractor all --pp_suffix $SUFFIX \
--times $NEW_TIME

# rouge
python -m data.evaluation --source_type api --time after0301 --source_dataset HC3 --task evaluate --times $NEW_TIME --pp_suffix $SUFFIX
