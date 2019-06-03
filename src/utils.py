import candle
import ignite.engine as engine
import ignite.metrics as ignite_metrics
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import torch
import tqdm


def create_train_model_fn(max_epochs, optimizer_fn, loss_fn, train_dataloader, eval_dataloader, example_batch_x, metrics=None):

    if metrics is None:
        metrics = {'mse': ignite_metrics.MeanSquaredError()}

    def train_model(model, name, history, device=None):
        model(example_batch_x)  # Initialise all layers in model before passing parameters to optimizer
        optimizer = optimizer_fn(model.parameters())

        history[name] = {'train log-loss': [], 'train mse': [], 'val log-loss': [], 'val mse': []}

        if device not in ('cuda', 'cpu'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        trainer = candle.create_supervised_trainer(model, optimizer, loss_fn, check_nan=True, grad_clip=1.0, device=device)
        evaluator = engine.create_supervised_evaluator(model, device=device, 
                                                       metrics={**metrics, 'log-loss': ignite_metrics.Loss(loss_fn)})
        
        log_interval = 10
        desc = "Epoch: {:4} Loss: {:5.3f}"
        num_batches = len(train_dataloader)
        
        @trainer.on(engine.Events.STARTED)
        def log_results(trainer):

            # training
            evaluator.run(train_dataloader)
            train_mse = evaluator.state.metrics['mse']
            train_loss = evaluator.state.metrics['log-loss']

            # testing
            evaluator.run(eval_dataloader)
            val_mse = evaluator.state.metrics['mse']
            val_loss = evaluator.state.metrics['log-loss']


            tqdm.tqdm.write("train mse: {:.5f} --- train log-loss: {:.1f} --- val mse: {:.5f} --- val log-loss: {:.1f}"
                            .format(train_mse, train_loss, val_mse, val_loss), file=sys.stdout)

            model_history = history[name]
            model_history['train mse'].append(train_mse)
            model_history['train log-loss'].append(train_loss)
            model_history['val mse'].append(val_mse)
            model_history['val log-loss'].append(val_loss)
        
        @trainer.on(engine.Events.EPOCH_STARTED)
        def create_pbar(trainer):
            trainer.state.pbar = tqdm.tqdm(initial=0, total=num_batches, desc=desc.format(0, 0), file=sys.stdout)

        @trainer.on(engine.Events.ITERATION_COMPLETED)
        def log_training_loss(trainer):
            iteration = (trainer.state.iteration - 1) % len(train_dataloader) + 1
            if iteration % log_interval == 0:
                trainer.state.pbar.desc = desc.format(trainer.state.epoch, trainer.state.output)
                trainer.state.pbar.update(log_interval)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def log_results_(trainer):
            trainer.state.pbar.n = num_batches
            trainer.state.pbar.last_print_n = num_batches
            trainer.state.pbar.refresh()        
            trainer.state.pbar.close()
            log_results(trainer)

        start = time.time()
        trainer.run(train_dataloader, max_epochs=max_epochs)
        end = time.time()
        tqdm.tqdm.write("Training took {:.2f} seconds.".format(end - start), file=sys.stdout)
        
    return train_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_results(signet, relunet, history):
    fig, axs = plt.subplots(2, 2, gridspec_kw={'wspace': 0.6, 'hspace': 0.6}, figsize=(7, 7))
    axs = axs.flatten()
    for i, metric_name in enumerate(('train mse', 'train log-loss', 'val mse', 'val log-loss')):
        ax = axs[i]
        signet_metric = history['SigNet'][metric_name]
        relunet_metric = history['ReluNet'][metric_name]
        if 'loss' in metric_name:
            signet_metric = np.log10(signet_metric)
            relunet_metric = np.log10(relunet_metric)
            metric_name = 'log-' + metric_name
        ax.plot(signet_metric, 'r-', lw=1, label='SigNet')
        ax.plot(relunet_metric, 'b-', lw=1, label='ReluNet')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
    plt.show()

    print(f'SigNet parameter count: {count_parameters(signet)}')
    print(f'ReluNet parameter count: {count_parameters(relunet)}')