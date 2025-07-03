from ner_evaluation import evaluate_ner_performance
from helpers import get_true_entities, get_labels


def evaluate(trained_model, test_dataset, threshold: float = 0.5, selected_labels: list | None = None) -> dict:
  """Evaluate the performance of a GLiNER model on a test dataset.

  Args:
    trained_model: The GLiNER model to evaluate.
    test_dataset: The test dataset to evaluate the model on.
    threshold: The threshold to use for entity prediction.
    selected_labels: The scpecific labels to evaluate. If None, all labels are evaluated.

  Returns:
    A dictionary containing the precision, recall, and F1 score for each evaluation method.
  """

  performances = {
    "exact": {"p": 0.0, "r": 0.0, "f1": 0.0},
    "relaxed": {"p": 0.0, "r": 0.0, "f1": 0.0}
  }
  
  true_entities = []
  predicted_entities = []
  
  """
  Go through test dataset, take example by example:
  - take true entities 
  - predict entites with GLiNER
  - compare true and predicted entities
  """
  for example in test_dataset:

    if selected_labels:
      ner = [ner for ner in example["ner"] if ner[2] in selected_labels]
    else:
      ner = example["ner"]

    # get text to find entities
    text = example["text"]
    
    # get labels for entity prediction
    labels = get_labels(ner)

    # predicted entities
    predicted_entities.append(trained_model.predict_entities(text, labels, threshold=threshold))
   
    # get true entities from "ner" column
    true_entities.append(get_true_entities(ner, example["tokenized_text"]))
  
  # evaluate the performance
    for match_type in performances:
        p, r, f1 = evaluate_ner_performance(true_entities, predicted_entities, match_type)
        performances[match_type]["p"] = p
        performances[match_type]["r"] = r
        performances[match_type]["f1"] = f1
  
  return performances