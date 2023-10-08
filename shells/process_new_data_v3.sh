# NEW_TIME="2023-04-17 2023-04-18 2023-04-19 2023-04-20 2023-04-21 2023-04-22 2023-04-23"
# NEW_TIME="2023-05-01 2023-05-02 2023-05-03 2023-05-04 2023-05-05 2023-05-06 2023-05-07 2023-05-08"
# NEW_TIME="2023-06-01 2023-06-02 2023-06-03 2023-06-04 2023-06-05 2023-06-06 2023-06-07 2023-06-08 2023-06-09 2023-06-10"
# NEW_TIME="2023-06-11 2023-06-12 2023-06-13 2023-06-14 2023-06-15 2023-06-16 2023-06-17 2023-06-18 2023-06-19 2023-06-20"
# NEW_TIME="2023-06-21 2023-06-22 2023-06-23 2023-06-24 2023-06-25 2023-06-26 2023-06-27 2023-06-28 2023-06-29 2023-06-30"
NEW_TIME="2023-08-02 2023-08-04"
SUFFIX="base1 base2 base3"

# knowledge, linguistic and classify features
CUDA_VISIBLE_DEVICES=7 python -m model.analyse --source_type api --source_dataset HC3 --time after0301 \
 --task extract_feature --end_id 1000 --feature_type all --knowledge_extractor all --pp_suffix $SUFFIX \
--times $NEW_TIME

# rouge
python -m data.evaluation --source_type api --time after0301 --source_dataset HC3 --task evaluate --times $NEW_TIME --pp_suffix $SUFFIX
