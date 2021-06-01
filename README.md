# Statistically Evaluating Social Media Sentiment Trends towards COVID-19 Non-Pharmaceutical Interventions with Event Studies

Code, analysis and data for the paper _Statistically Evaluating Social Media Sentiment Trends towards COVID-19 Non-Pharmaceutical Interventions with Event Studies_.

In the midst of a global pandemic, understanding the public's opinion of their government's policy-level, non-pharmaceutical interventions (NPIs) is a crucial component of the health-policy-making process. Prior work on CoViD-19 NPI sentiment analysis by the epidemiological community has proceeded without a method for properly attributing sentiment changes to events, an ability to distinguish the influence of various events across time, a coherent model for predicting the public's opinion of future events of the same sort, nor even a means of conducting significance tests. We argue here that this urgently needed evaluation method does already exist. In the financial sector, event studies of the fluctuations in a publicly traded company's stock price are commonplace for determining the effects of earnings announcements, product placements, etc. The same method is suitable for analysing temporal sentiment variation in the light of policy-level NPIs. We provide a case study of Twitter sentiment towards policy-level NPIs in Canada. Our results confirm a generally positive connection between the announcements of NPIs and Twitter sentiment, and we document a promising correlation between the results of this study and a public-health survey of popular compliance with NPIs.

# Quick Start

## Event Study of COVID NPI Sentiments:
```sh
python analysis/studies.py
```

## CAR & Survey Correlation
```sh
python analysis/correlation.py
```

# Citation

```BibTeX
@inproceedings{niu-etal-2021-statistically,
    title = "Statistically Evaluating Social Media Sentiment Trends towards {COVID}-19 Non-Pharmaceutical Interventions with Event Studies",
    author = "Niu, Jingcheng  and
      Rees, Erin  and
      Ng, Victoria  and
      Penn, Gerald",
    booktitle = "Proceedings of the Sixth Social Media Mining for Health ({\#}SMM4H) Workshop and Shared Task",
    month = jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.smm4h-1.1",
    pages = "1--6",
    abstract = "In the midst of a global pandemic, understanding the public{'}s opinion of their government{'}s policy-level, non-pharmaceutical interventions (NPIs) is a crucial component of the health-policy-making process. Prior work on CoViD-19 NPI sentiment analysis by the epidemiological community has proceeded without a method for properly attributing sentiment changes to events, an ability to distinguish the influence of various events across time, a coherent model for predicting the public{'}s opinion of future events of the same sort, nor even a means of conducting significance tests. We argue here that this urgently needed evaluation method does already exist. In the financial sector, event studies of the fluctuations in a publicly traded company{'}s stock price are commonplace for determining the effects of earnings announcements, product placements, etc. The same method is suitable for analysing temporal sentiment variation in the light of policy-level NPIs. We provide a case study of Twitter sentiment towards policy-level NPIs in Canada. Our results confirm a generally positive connection between the announcements of NPIs and Twitter sentiment, and we document a promising correlation between the results of this study and a public-health survey of popular compliance with NPIs.",
}

```