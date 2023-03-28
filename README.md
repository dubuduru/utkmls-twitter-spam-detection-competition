# utkmls-twitter-spam-detection-competition
Kaggle competition for twitter spam detection: datail [HERE](https://www.kaggle.com/competitions/utkmls-twitter-spam-detection-competition/overview)

## Method
Huggingface 내의 크고 작은 모델을 fine-tune해보며 최적의 모델을 찾고자 하였습니다.
AdamW optimizer의 beta값과 LR 등의 hyperparameter를 조정해보며 실험하였습니다.

### Obstacles
Original GPT 논문의 LR (2.5e-4)을 이용하여 bert-base-uncased 및 distilbert를 학습시킬 경우 accuracy가 계속 낮아지는 현상이 있었습니다.
(작은 모델인 bert-tiny등에서는 해당 LR이 잘 작동함을 확인하였습니다)
발산하는 현상으로 판단하여 LR을 1e-5로 낮추어 진행하였고, 잘 학습됨을 확인하였습니다.

## Dependency
본 코드는 [Kaggle docker 환경](https://github.com/kaggle/docker-python) 위에서 작동합니다.

## Log
<details>
<summary>bert-base-uncased fine-tuning (2 epochs)</summary>
<div markdown="1">

<img width="644" alt="스크린샷 2023-03-28 오후 11 15 37" src="https://user-images.githubusercontent.com/78554126/228274080-6b309f9f-141a-4899-84d9-f7366e6cce32.png">
  
<img width="715" alt="스크린샷 2023-03-28 오후 11 31 52" src="https://user-images.githubusercontent.com/78554126/228274288-b2685a8a-a85a-4057-b23a-6c6c45029afd.png">

</div>
</details>

<details>
<summary>bert-base-uncased validation</summary>
<div markdown="1">

<img width="495" alt="스크린샷 2023-03-28 오후 11 31 59" src="https://user-images.githubusercontent.com/78554126/228274871-e2fcd061-603e-497d-a6a1-a45f305b88c6.png">

</div>
</details>

</div>
</details>

<details>
<summary>submission</summary>
<div markdown="1">

<img width="1111" alt="스크린샷 2023-03-28 오후 11 34 40" src="https://user-images.githubusercontent.com/78554126/228275110-203d3a01-bca6-457a-9b9a-17c98098aa10.png">

</div>
</details>

## Bonus (Optional)
Prompting 관련 아티클을 읽어본 결과, ChatGPT에 프롬프트를 넣어줄 때에는 "I want you to act as ~"로 시작하는 것이 좋다고 합니다. 상황 설명 후, 해당 Kaggle competetion에서 정의한 spam의 개념을 입력합니다.

In-context learning을 위해 8개의 example을 넣어주었습니다. example의 순서에서 어떠한 bias를 얻을 수도 있다고 생각하여, spam과 non-spam을 번갈아가며 넣어주었습니다. spam에 해당하는 4개의 example은 각각 kaggle에서 정의한 4가지 spam feature에 매칭되도록 선정하였습니다.

또한 Input을 넣을 때에는 3개의 큰따옴표로 씌워주는 것이 좋다고 하여, 입력 sentence를 `""" [sentence] """` 형식으로 구성하였습니다.

출력은 0(non-spam) or 1(spam)입니다.
