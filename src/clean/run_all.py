import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, "..", "..", "dataset")
OUT_DIR = os.path.join(BASE, "..", "..", "data")
sys.path.insert(0, BASE)


def path(subdir, name):
    return os.path.join(DATASET, subdir, name)


def main():
    import clean_recorder
    import clean_company_context
    import clean_benchmark_bands
    import clean_client_cover_summary
    import clean_synthetic_briefs
    import clean_recommendations
    import clean_deck_current_cover
    import clean_deck_limit_benchmarks
    import assert_join_integrity

    clean_recorder.run(
        path("2_book_of_business", "recorder_book_of_business"),
        path("2_book_of_business", "recorder_cover_lines"),
        out_dir=OUT_DIR,
    )
    clean_company_context.run(
        path("3_company_context", "healthcheck_questionnaire"),
        path("3_company_context", "hubspot_company_context"),
        out_dir=OUT_DIR,
    )
    clean_benchmark_bands.run(path("2_book_of_business", "deck_benchmark_bands"), out_dir=OUT_DIR)
    clean_client_cover_summary.run(path("2_book_of_business", "recorder_cover_lines"), out_dir=OUT_DIR)
    clean_synthetic_briefs.run(path("4_evaluation", "synthetic_prospect_briefs"), out_dir=OUT_DIR)
    clean_recommendations.run(path("2_book_of_business", "deck_recommendations"), out_dir=OUT_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)

    cc = clean_deck_current_cover.parse_dataframe(
        clean_deck_current_cover.load(path("2_book_of_business", "deck_current_cover"))
    )
    cc.to_csv(os.path.join(OUT_DIR, "deck_current_cover_parsed.csv"), index=False)

    lb = clean_deck_limit_benchmarks.parse_dataframe(
        clean_deck_limit_benchmarks.load(path("2_book_of_business", "deck_limit_benchmarks"))
    )
    lb.to_csv(os.path.join(OUT_DIR, "deck_limit_benchmarks_parsed.csv"), index=False)

    print()
    assert_join_integrity.run(data_dir=OUT_DIR)


if __name__ == "__main__":
    main()