# NEW_TIME="2023-04-17 2023-04-18 2023-04-19 2023-04-20 2023-04-21 2023-04-22 2023-04-23"
# NEW_TIME="2023-08-11 2023-08-12 2023-08-13 2023-08-14 2023-08-15 2023-08-16 2023-08-17 2023-08-18 2023-08-19 2023-08-20"
NEW_TIME="2023-08-21 2023-08-22 2023-08-23 2023-08-24 2023-08-25 2023-08-26"
# NEW_TIME="2023-08-11 2023-08-12 2023-08-13 2023-08-14 2023-08-15 2023-08-16 2023-08-17 2023-08-18 2023-08-19 2023-08-20"
SUFFIX="base1 base2 base3"

# knowledge, linguistic and classify features
CUDA_VISIBLE_DEVICES=6 python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
 --task extract_feature --end_id 1000 --feature_type all --knowledge_extractor all --pp_suffix $SUFFIX \
--times $NEW_TIME

# rouge
python -m data.evaluation --source_type api --time after0301 --source_dataset HC3 --task evaluate --times $NEW_TIME --pp_suffix $SUFFIX
