# 2023-08-11 2023-08-12 2023-08-13 2023-08-14 2023-08-15 2023-08-16 
NEW_TIME="2023-08-11 2023-08-12 2023-08-13 2023-08-14 2023-08-15 2023-08-16 2023-09-09"
SUFFIX="base1 base2 base3"


python -m data.evaluation --source_type api --time after0301 --source_dataset HC3 --task evaluate --times $NEW_TIME --pp_suffix $SUFFIX



