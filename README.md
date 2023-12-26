# PATHVQA: Pathology Visual Question Answering Dataset

## Overview

PATHVQA is a specialized dataset tailored for training and evaluating AI models in pathology visual question answering (VQA). With a focus on overcoming challenges in medical dataset creation, PATHVQA is designed to enhance AI applications for pathology. The PATHVQA dataset is publicly released to contribute to research in medical visual question answering. It aims to support the academic community in fostering advancements in AI-driven clinical decision support and medical education.

## Key features 
### Size: 
Comprising 4,998 pathology images from freely accessible textbooks and online libraries.
### Diversity: 
Offers 32,799 open-ended questions, mirroring the complexity of the American Board of Pathology examination.
### Privacy Assurance: 
The dataset creation process prioritizes the anonymity and confidentiality of medical information, addressing privacy concerns.


## Requirements

To run the code in the Jupyter Notebook, you'll need the following Python libraries:

- `PIL`
- `numpy`
- `pandas`
- `nltk`
- `os`
- `tqdm`
- `matplotlib`
- `torch`
- `torchtext`
- `torchvision`

You can install these dependencies using the following command:

```bash
pip install pillow numpy pandas nltk tqdm matplotlib torch torchtext torchvision
```

Additionally, make sure you have Jupyter Notebook installed:
```bash
!pip install jupyter
 ```

## Training

To train the model, use the provided `train_model` function in your Jupyter Notebook. The function takes the following parameters:

- `model`: The neural network model you want to train.
- `criterion`: The loss function, in this case, it's `nn.CrossEntropyLoss`.
- `optimizer`: The optimization algorithm, in this case, it's stochastic gradient descent (SGD).
- `num_epochs`: The number of epochs for training.

Here's an example of how to use the function:

```python
# Import necessary libraries
# Define your model, criterion, and optimizer
torch.manual_seed(42)
model = VQAModel(input_representation_shape=512, txt_embedd_dim=EMBEDDING_DIM, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.9)

# Train the model for one epoch
num_epochs = 1
model_trained = train_model(model, criterion, optimizer, num_epochs=num_epochs)
```
## Evaluation

To evaluate the trained model on a test set and calculate the F1 score, use the following code in your Jupyter Notebook:

```python
# Import necessary libraries
# Set the model to evaluation mode
model_trained.eval()

# Initialize an empty list to store F1 scores
f1_scores = []

# Use the model to make predictions on the test set
with tqdm(dataloaders['test'], unit='batch', position=0, leave=True) as pbar:
    for X, y, z in pbar:
        pbar.set_description("Evaluating")

        X = X.to(device)
        y = y.to(device)
        z = z.to(device)

        # Forward pass
        outputs = model_trained(X, y)
        _, preds = torch.max(outputs, 1)

        # Calculate F1 score and append to the list
        f1_scores.append(f1_score(np.array(z.cpu()), np.array(preds.cpu()), average='micro'))

        pbar.set_postfix(F1_Score=np.mean(np.array(f1_scores)))
        sleep(0.1)

# Calculate the overall F1 score
overall_f1_score = np.mean(np.array(f1_scores))
print('Overall F1 Score: {:.4f}'.format(overall_f1_score))
```










