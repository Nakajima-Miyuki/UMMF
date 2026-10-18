# UMMF

Code for the paper on Unlearning-based Multimodal Memorization Fingerprint (UMMF), an implicit fingerprinting method for copyright authentication of Large Vision-Language Models (LVLMs).

UMMF strengthens the memorization of fingerprint samples by learning the original caption while unlearning a semantically similar neighboring caption of the same image. Ownership is verified through difficulty-calibrated membership inference (Min K%) over the neighboring sample pairs.

## Dependency

+ python 3.9+
+ torch 2.2.0+
+ transformers 4.41.2+
+ numpy 1.26.4+
+ scikit-learn 1.3.0+
+ peft 0.11.0+

## Dataset & Models

+ Base model: LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`)
+ Fingerprint / reference samples: 100 + 100 samples sampled from Flickr30k
+ Paraphrase generation: Qwen2.5-VL-7B-Instruct
+ Downstream fine-tuning: V7W, ST-VQA, TextVQA, PaintingForm, MathV360k, ChEBI-20

## Citation

```bibtex
@inproceedings{zheng-etal-2026-ummf,
    title = "{UMMF}: Protecting Copyright of Large Vision-Language Models through Unlearning-based Multimodal Memorization Fingerprint",
    author = "Zheng, Xiaofan  and
      Wang, Xinghao  and
      Wan, Xiaojun",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.429/",
    doi = "10.18653/v1/2026.acl-long.429",
    pages = "9497--9513",
    ISBN = "979-8-89176-390-6",
    abstract = "Training Large Vision-Language Models (LVLMs) is costly and resource-intensive, making them valuable assets. To prevent malicious users from unauthorized commercialization of these artificial intelligence assets through fine-tuning and black-box deployment, model fingerprinting techniques aimed at verifying the ownership of LVLMs are receiving widespread attention. Existing fingerprinting techniques rely on adversarial attacks or backdoor attacks to construct trigger images for specific outputs, attributing model ownership by comparing whether the output of trigger images on suspected models matches the predetermined output. However, these methods depend on fixed-form triggers as explicit model fingerprints, which have limitations in terms of stealthiness and robustness. Inspired by unlearning research, we propose Unlearning-based Multimodal Memorization Fingerprint (UMMF). UMMF strengthens the overfitting characteristics of training samples by unlearning neighboring samples of the training samples, thereby introducing detectable regions of poor generalization in the data manifold. Compared with previous methods, our approach leverages the differences in memorization strength of LVLMs on neighboring samples as implicit model fingerprints, rather than relying on specific input-output pairs as explicit triggers. This endows it with stronger stealthiness, robustness, and adaptability. To simulate real application scenarios, we conduct extensive experiments using multiple strategies and different datasets, further demonstrating its superiority in protecting LVLM ownership."
}
```
