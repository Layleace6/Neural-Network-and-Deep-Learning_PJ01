import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This class is designed to train, evaluate, save, and load a model. 
    It has been modified so that evaluation is performed once per epoch instead of after every iteration.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=128, scheduler=None, regularization=None):
        """
        Args:
            model: The neural network model to train.
            optimizer: The optimization algorithm (e.g., SGD).
            metric: The evaluation metric (e.g., accuracy).
            loss_fn: The loss function (e.g., CrossEntropyLoss).
            batch_size: Batch size for training.
            scheduler: Learning rate scheduler (optional).
            regularization: Regularization term (e.g., L2Regularization, optional).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.regularization = regularization  # Add regularization term

        # Lists to store training and validation metrics
        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        """
        Train the model for a given number of epochs and evaluate once per epoch.
        """
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        # Create directory to save the best model if it doesn't exist
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0  # To track the best validation score

        for epoch in range(num_epochs):
            self.model.train()

            X, y = train_set
            assert X.shape[0] == y.shape[0]
            idx = np.random.permutation(range(X.shape[0]))  # Shuffle data
            X = X[idx]
            y = y[idx]

            num_batches = X.shape[0] // self.batch_size

            # tqdm progress bar for tracking progress within an epoch
            with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                epoch_train_loss = 0.0
                epoch_train_score = 0.0

                for iteration in range(num_batches):
                    train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                    train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                    logits = self.model(train_X)
                    
                    # Compute main loss
                    trn_loss = self.loss_fn(logits, train_y)
                    
                    # Add regularization loss if provided
                    if self.regularization is not None:
                        reg_loss = self.regularization()  # Get regularization loss
                        trn_loss += reg_loss  # Add regularization loss to total loss
                    
                    epoch_train_loss += trn_loss
                    
                    trn_score = self.metric(logits, train_y)
                    epoch_train_score += trn_score

                    # Backward pass for main loss
                    self.loss_fn.backward()
                    
                    # Add regularization gradients if provided
                    if self.regularization is not None:
                        self.regularization.backward()  # Add regularization gradients

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Update tqdm progress bar description
                    pbar.set_postfix({"train_loss": f"{trn_loss:.4f}", "train_score": f"{trn_score:.4f}"})
                    pbar.update(1)

                # Average the training loss and score for the epoch
                avg_train_loss = epoch_train_loss / num_batches
                avg_train_score = epoch_train_score / num_batches
                self.train_loss.append(avg_train_loss)
                self.train_scores.append(avg_train_score)

            # Evaluate the model on the validation set at the end of the epoch
            dev_score, dev_loss = self.evaluate(dev_set)
            self.dev_scores.append(dev_score)
            self.dev_loss.append(dev_loss)

            # Log the results for the epoch
            print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train Score: {avg_train_score:.4f} | "
                  f"Dev Loss: {dev_loss:.4f}, Dev Score: {dev_score:.4f}")

            # Save the model if the validation score improves
            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"Best accuracy updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score

        self.best_score = best_score

    def evaluate(self, data_set):
        """
        Evaluate the model on the given dataset.
        """
        self.model.eval()
        X, y = data_set
        eval_batch_size = 2048
        assert X.shape[0] == y.shape[0]
        num_batches = X.shape[0] // eval_batch_size

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i in range(num_batches):
            batch_X = X[i * eval_batch_size : (i + 1) * eval_batch_size]
            batch_y = y[i * eval_batch_size : (i + 1) * eval_batch_size]

            logits = self.model(batch_X)
            loss = self.loss_fn(logits, batch_y)

            preds = np.argmax(logits, axis=1)
            correct = np.sum(preds == batch_y)

            total_loss += loss * batch_X.shape[0]
            total_correct += correct
            total_samples += batch_X.shape[0]

        final_loss = total_loss / total_samples
        final_score = total_correct / total_samples  # Accuracy
        return final_score, final_loss

    def save_model(self, save_path):
        """
        Save the model to the specified path.
        """
        self.model.save_model(save_path)