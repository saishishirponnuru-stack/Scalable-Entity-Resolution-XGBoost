import pandas as pd
import re
import os


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r'[^\w\s]',
        ' ',
        text
    )

    replacements = {
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\bltd\b': 'limited',
        r'\bpvt\b': 'private',
        r'\bco\b': 'company',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard'
    }

    for pattern, replacement in replacements.items():

        text = re.sub(
            pattern,
            replacement,
            text
        )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


def preprocess_and_save(
    file_path,
    output_file
):

    print(
        f"Scrubbing {file_path}..."
    )

    df = pd.read_csv(
        file_path,
        sep='\t'
    )

    df['clean_name'] = (
        df['business_name']
        .apply(clean_text)
    )

    df['clean_address'] = (
        df['business_address']
        .apply(clean_text)
    )

    df['clean_country'] = (
        df['country']
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Saving to {output_file}..."
    )

    df.to_csv(
        output_file,
        sep='\t',
        index=False
    )


# ============================================================
# TRAIN
# ============================================================

os.makedirs(
    'clean_train_data',
    exist_ok=True
)

preprocess_and_save(
    'train/train_source1.tsv',
    'clean_train_data/clean_source1.tsv'
)

preprocess_and_save(
    'train/train_source2.tsv',
    'clean_train_data/clean_source2.tsv'
)

preprocess_and_save(
    'train/train_source3.tsv',
    'clean_train_data/clean_source3.tsv'
)


# ============================================================
# TEST
# ============================================================

os.makedirs(
    'clean_test_data',
    exist_ok=True
)

preprocess_and_save(
    'test/test_source1.tsv',
    'clean_test_data/clean_source1.tsv'
)

preprocess_and_save(
    'test/test_source2.tsv',
    'clean_test_data/clean_source2.tsv'
)

preprocess_and_save(
    'test/test_source3.tsv',
    'clean_test_data/clean_source3.tsv'
)


print("\n" + "=" * 70)
print("TRAIN + TEST SCRUBBING COMPLETE")
print("=" * 70)