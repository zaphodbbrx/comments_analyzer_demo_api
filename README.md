# comments_analyzer_demo_api

Modified version of https://github.com/zaphodbbrx/category_classifier_demo . The difference is in automatic language detection and output format which is now a dictionary with the following fields:

* comment - actual comment that was analyzed

* lang - detected language

* root category - main category

* probability - probability of prediction (generated via predict_proba method)

* subcategory - subcategoory if available (i.e. only for balance at the moment)
