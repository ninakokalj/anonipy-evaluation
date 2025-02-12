from ner_evaluation import evaluate_ner_performance
from helpers import get_true_ents, get_labels


def evaluate(trained_model, test_dataset, threshold: float = 0.5) -> dict:
  """Evaluate the performance of a GLiNER model on a test dataset."""

  performances = {
    "exact": {"p": 0.0, "r": 0.0, "f1": 0.0},
    "relaxed": {"p": 0.0, "r": 0.0, "f1": 0.0}
  }
  
  true_ents = []
  pred_ents = []
  
  for example in test_dataset:
    # get text to find entities
    text = example["text"]
    
    # get labels for entity prediction
    labels = get_labels(example["ner"])

    # Predicted entities
    pred_ents.append(trained_model.predict_entities(text, labels, threshold=threshold))
   
    # get true entities from "ner" column
    true_ents.append(get_true_ents(example["ner"], example["tokenized_text"]))
  
  # evaluate the performance
    for match_type in performances:
        p, r, f1 = evaluate_ner_performance(true_ents, pred_ents, match_type)
        performances[match_type]["p"] = p
        performances[match_type]["r"] = r
        performances[match_type]["f1"] = f1
  

  return performances






