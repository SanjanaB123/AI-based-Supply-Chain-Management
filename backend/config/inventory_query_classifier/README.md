---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: What's the weather today?
- text: Which items are low on stock?
- text: What is the total stock?
- text: What's the meaning of life?
- text: Show me the total units sold per store
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [SetFitHead](huggingface.co/docs/setfit/reference/main#setfit.SetFitHead) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [SetFitHead](huggingface.co/docs/setfit/reference/main#setfit.SetFitHead) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 3 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                              |
|:------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | <ul><li>'What is the total stock?'</li><li>'Show me the inventory summary'</li><li>'How many products do we have in Store A?'</li></ul>                               |
| 1     | <ul><li>'Predict future sales for the next month'</li><li>'Forecast inventory needs for next week'</li><li>'What will be the stock levels after Christmas?'</li></ul> |
| 2     | <ul><li>"What's the weather today?"</li><li>'Tell me a joke.'</li><li>'How do I bake a cake?'</li></ul>                                                               |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("What is the total stock?")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 3   | 6.1    | 9   |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 10                    |
| 1     | 10                    |
| 2     | 10                    |

### Training Hyperparameters
- batch_size: (8, 8)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0133 | 1    | 0.4628        | -               |
| 0.6667 | 50   | 0.1505        | -               |

### Framework Versions
- Python: 3.11.9
- SetFit: 1.1.3
- Sentence Transformers: 5.3.0
- Transformers: 4.45.2
- PyTorch: 2.11.0
- Datasets: 4.8.4
- Tokenizers: 0.20.3

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->