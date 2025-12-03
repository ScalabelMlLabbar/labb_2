---
title: Iris
emoji: 🐨
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: apache-2.0
---


Describe in your README.md program ways in which you can improve
model performance are using

### (a) model-centric approach

Some of the hyperparameters specific to LoRA adapters are rank and target modules. The rank in LoRA controls the number of trainable parameters in the adapter matrices. Currently, we are using a rank of 16. A higher rank entails a larger number of parameters being updated, allowing the model to capture more nuanced patterns in the data, but it requires more memory. A smaller rank results in faster fine-tuning but with fewer trainable parameters and potentially lower expressiveness. The target modules are the specific parts of the model that we apply the LoRA adapters to. We applied the LoRA adapters to both the attention and MLP layers, meaning we fine-tune both the attention layers (for context understanding) and MLP. Applying LoRA to only one type of module would reduce memory usage but might decrease model performance.

Learning rate is an important hyperparameter. A higher learning rate allows the model to converge faster initially, but too high a rate can cause instabilities or failure to find the global optimum. Conversely, a too-low learning rate requires more epochs to converge, increasing training time and potentially preventing sufficient learning. In our model, we set the learning rate to 2e-4, as recommended in the Unsloth guide for LoRA fine-tuning [1]. However, the ideal way would have been to start at the recommended level, then try different learning rates and compared them to find the optimal result. 

The number of epochs also affects performance. We chose 1 epoch for our models to minimize training time, as GPU resources were limited. However, the ideal would be to train for multiple epochs to allow the model to fully learn patterns in the dataset, refine its weights, improve generalization, while monitoring for overfitting.

The effective batch size affects training stability and model quality. The effective batch size is calculated as “batch size × gradient accumulation steps”. A higher batch size improves gradient stability but requires more GPU memory, while a smaller batch size reduces memory usage but may lead to noisier updates and slower convergence.


### (b) data-centric approach

To get better results, on our domain task, gen Z slang, we can shift our focus from just tweaking the model's settings to actually improving the data it learns from. This is a datacentric aprotche. The problem right now is that standard models are trained on formal, static text, so they often just guess when they encounter real-world slang. By finding and feeding the model fresh, raw datasets that capture these more casual areas. This can help the model performance on this specific task.

### References
[1] Unsloth. "LoRA Hyperparameters Guide." Unsloth Documentation, https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide. Accessed 3 Dec. 2025.

