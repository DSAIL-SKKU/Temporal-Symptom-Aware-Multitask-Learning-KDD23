## Abstract
Bipolar disorder (BD) is closely associated with an increased risk of suicide. However, while the prior work has revealed valuable insight into understanding the behavior of BD patients on social media, little attention has been paid to developing a model that can predict the future suicidality of a BD patient. This study proposes a multi-task learning model for predicting the future suicidality of BD patients by jointly learning current symptoms. We build a novel BD dataset clinically validated by psychiatrists, including 14 years of posts on bipolar-related subreddits by 818 BD patients, along with the annotations of future suicidality and BD symptoms. We also suggest a multi-task learning model that applies a temporal symptom-aware attention mechanism to determine which symptoms are the most influential for predicting future suicidality over time through a sequence of BD posts. The proposed model outperforms the state-of-the-art models in both BD symptom identification and future suicidality prediction tasks. Additionally, the proposed temporal symptom-aware attention provides interpretable attention weights, helping clinicians to apprehend BD patients more comprehensively and to provide timely intervention by tracking mental state progression. 
![](https://lh5.googleusercontent.com/qGH3ow3-dv-GhN1GEJ4CU59EfnfNI9xCSI5vyaU8SvoHEL5-xKXL-9YC2iFqGDJFTUCurqRROIJ8lUFEbLcc_OnKBbDkwypcerQXagaV0U0onfUY74QNfRVwSAtukYA0OQ=w1280)

## Annotation Process
To label the collected Reddit dataset, we recruited four researchers, who are knowledgeable in psychology and fluent in English, as annotators. With the supervision of a psychiatrist, the four trained annotators labeled 818 users and their 7,592 anonymized Reddit posts using the open-source text annotation tool Doccano. During annotations, we mainly consider two different label categories: (i) BD symptoms (e.g., manic, anxiety) and (ii) suicidality levels (e.g., ideation, attempt). We further annotate the diagnosed BD type (e.g., BD-I, BD-II) for data analysis. If there is any conflict in the annotated labels across the annotators, all the annotators discuss and reach to an agreement under the supervision of the psychiatrists. 
![](https://lh4.googleusercontent.com/P5sjfdhX1NtGY5bzTk4jFCs3nHnA00C236URgrN3JWTg-uhRGtLhuY5yVnYBzS34qlhM1sd3cQ_U2NZTs3_9658twhppFsouQ-Q_xhYJGXJHpMHImjw3Cc_wbX0iS6wm1Q=w1280)

## Ethical Concerns
We carefully consider potential ethical issues in this work: (i) protecting users' privacies on Reddit and (ii) avoiding potentially harmful uses of the proposed dataset. The Reddit privacy policy explicitly authorizes third parties to copy user content through the Reddit API. We follow the widely-accepted social media research ethics policies that allow researchers to utilize user data without explicit consent if anonymity is protected (benton et al. 2017; Williams et al., 2017). Any metadata that could be used to specify the author was not collected. In addition, all content is manually scanned to remove personally identifiable information and mask all the named entities. More importantly, the BD dataset will be shared only with other researchers who have agreed to the ethical use of the dataset. This study was reviewed and approved by the Institutional Review Board (SKKU2022-11-038).

## How to Request Access

While it is important to ensure that all necessary precautions are taken, we are enthusiastic about sharing this valuable resource with fellow researchers. To request access to the dataset, please contact Daeun Lee (delee12@skku.edu). Access requests should follow the format of the sample application provided below, which consists of three parts:

Part 0: Download a sample application form (https://sites.google.com/view/daeun-lee/dataset/kdd-2023)
Part 1: Applicant Information
Part 2: Dataset Access Application
Part 3: Ethical Review by Your Organization

The dataset was produced at Sungkyunkwan University (SKKU) in South Korea, and the research conducted on this dataset at SKKU has been granted exemption from Institutional Review Board (IRB) evaluation by SKKU's IRB (SKKU2022-11-038). This exemption applies to the analysis of pre-existing data that is publicly accessible or involves individuals who cannot be directly identified or linked to identifiable information. Nevertheless, due to the potentially sensitive nature of this data, we require that researchers who receive the data obtain ethical approval from their respective organizations.

Please submit your access request to Daeun Lee (delee12@skku.edu) and ensure that you include all the necessary information and address the points outlined in the sample application.


## Dataset Availability and Governance Plan
Inspired by the data sharing system of previous research (Zirikly et al. 2019), we have decided to establish a governance process for researcher access to the dataset, following the procedure outlined below.
Due to limitations in the number of available individuals, three out of the five authors will be selected to review access requests submitted in the format specified below. The outcomes of the review will result in the following responses:

- Approval: If all three members give their approval, the application will be deemed approved, and Daeun will proceed to share the dataset with the researcher.
- Inquiries: The authors may have questions or seek clarification, prompting further communication.
- Revision and resubmission: Should the authors provide specific suggestions for revising and resubmitting the application, the researcher will have the opportunity to address them.
- Rejection: In the event of unanimous disapproval from the authors, the dataset will not be shared.

The authors will prioritize and promote diversity and inclusivity among the reviewers and the community of researchers utilizing the dataset.

Reference
Zirikly, A., Resnik, P., Uzuner, O., & Hollingshead, K. (2019, June). CLPsych 2019 shared task: Predicting the degree of suicide risk in Reddit posts. In Proceedings of the sixth workshop on computational linguistics and clinical psychology (pp. 24-33)# Temporal-Symptom-Aware-Multitask-Learning-KDD23
