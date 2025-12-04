---
title: Iris
emoji: 🐨
colorFrom: green
colorTo: pink
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
license: apache-2.0
---
# Task 2

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

----------------------------------------------------------
# Choosen LLM

### Choosen model: Llama 3.2 1B 

We worked witht three different instances. We initalied fintuned Llama 3.2 3B with the mlabonne/FineTome-100k, then we tried a smaller model, Llama 3.2 1B with the same dataset. Lastly we tried the Llama 3.2 3B again but fintuned with the dataset MLBtrio/genz-slang-dataset.

In order to evaluate them we tried two different prompts and compared the recived reposnses and time it took to generate. The two prompts we used were:
1. What is the meaning of this sentence, “Justin got cancelled yesterday, no cap. He was at a party clout chasing with no drip, no rizz and acting hella pressed! It’s giving red flag.”
2. What is the best way to study and gain knowledge about machine learning in a fast way?

We want to try if the finetuned model using the gen-z slang would perform better than the other two models. Therefore, we choose the first prompt. We created the second prompt as a "general question" as all models should be able to respond to a variety of queries.

### Results

Llama 3.2 1B FineTome response to prompt 1: 

This sentence describes Justin as being "cancelled" - which means he has lost credibility or reputation after speaking out against something. The phrase "no cap" implies that his comments were unfiltered and honest.
The second part of the sentence refers to how he felt about the situation, describing him as "at a party clout chasing with no drip", which suggests he had an inflated sense of self-importance. He also acted "hella pressed," implying he wanted attention for himself.
The third part of the sentence uses "red flags" to describe Justin's behavior. A red flag in a relationship usually indicates warning signs of potential trouble ahead. In this context, it could imply that there are certain behaviors or actions Justin may be taking that could lead to problems down the line. 



Llama 3.2 3B FineTome response to prompt 1: 

Let me break it down:

“Justin got cancelled” means Justin was publicly humiliated or embarrassed.
“No cap” is an expression that means “no lie” or “no exaggeration,” indicating that what's being said is true.
“Clout chasing” refers to someone trying to gain popularity or attention on social media by doing provocative or sensational things.
“Drip” is a slang term for style, especially fashion sense.
“Rizz” is a slang term for charisma or charm.
“Acting hella pressed” means someone is being extremely anxious or stressed out.
“Giving red flag” is a metaphorical expression that suggests something is suspicious or warning sign.
So, the entire sentence means that Justin was involved in a scandalous situation where he tried to appear fashionable and popular (clout chasing), but his attempts were unsuccessful, making him look ridiculous or embarrassing (hella pressed). The phrase is essentially saying that Justin's behavior raised suspicions and indicated that there might be deeper issues.


Llama 3.2 3B gen-z response to prompt 1: 

This is a A term used on social media to describe when someone or a group of people are called out for behavior considered to be morally wrong or unethical.. The concept of "cancellation" has evolved online to mean that an individual or organization is held accountable for their actions, often in response to public outcry.. Meaning that something is so bad it should be avoided.. Red flag indicates a potential warning sign of problems in a relationship.. Red flags can appear in various aspects of life, including relationships, business practices, and societal behaviors. It's often associated with red flags in dating, indicating potential issues down the line. In modern slang, "red flags" have taken on a new meaning related to dating and relationships, representing warning signs that something might go wrong.. Red flags in dating refer to behaviors or characteristics that may indicate trouble ahead. Justin Timberlake's song 'Say Say Sayberry' isn't related to this discussion either.


Llama 3.2 1B FineTome response to prompt 2:

There are several ways you can quickly learn about machine learning:

- Online Courses: Websites like Coursera, edX, and Udacity offer courses on various topics related to machine learning, such as deep learning, natural language processing, and more.
- Books and eBooks: There are many books available that cover machine learning concepts, including 'Machine Learning' by Andrew Ng and 'Deep Learning' by Ian Goodfellow and Yoshua Bengio and Aaron Coursera Specialization in Machine Learning by Stanford University.
- YouTube Tutorials and Videos: Channels like Machine Learning Mastery, Deep Dive Machine Learning, and Machine Learning Mastery have numerous video tutorials covering different aspects of machine learning.
- Practice Problems and Projects: Platforms like Kaggle, GitHub, and Google Colab provide free access to datasets and code examples for practicing machine learning algorithms.
- Join Online Communities: Participating in online forums like Reddit's Machine Learning community or Stack Overflow can help you connect with other learners and experts who share your interests.
- Attend Webinars and Conferences: These events often feature presentations and discussions on the latest advancements in machine learning and its applications.
- Take Self-Study Materials: Books, articles, and research papers can give you a deeper understanding of the subject matter.

Remember, while speed is important, consistency is key when it comes to mastering a topic. Focus on understanding the fundamentals first, then move towards practical applications or hands-on projects. 


Llama 3.2 3B FineTome response to prompt 2:

1. Online Courses:
  - Coursera, edX, Udacity, and Stanford CS231n are excellent platforms for learning machine learning.
  - Stanford CS231n is particularly recommended as it covers both supervised and unsupervised learning techniques.
2. Books:
  - "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville provides an overview of deep neural networks.
  - "Pattern Recognition and Machine Learning" by Christopher M. Bishop offers insights into various aspects of machine learning.
  - "Python Machine Learning" by Sebastian Raschka offers practical Python examples for beginners.
3. Practice:
  - Start with simple projects like image classification or sentiment analysis using libraries like TensorFlow, PyTorch, or Keras.
  - Use Kaggle competitions and datasets from UCI Machine Learning Repository for practice.
4. Communities:
  - Join online communities like Reddit's r/MachineLearning and Stack Overflow's Machine Learning tag for discussions and resources.
  - Participate in forums related to machine learning to learn from others' experiences and gain insights.
5. Podcasts:
  - Listen to podcasts like Data Science Weekly, Machine Learning Podcast, and AI Alignment Podcast to stay updated on industry developments and research advancements.
6. Conferences:
  - Attend conferences like NIPS (Neural Information Processing Systems), ICML (International Conference on Machine Learning), and IJCAI (International Joint Conference on Artificial Intelligence) to network with experts and learn about new breakthroughs.
7. YouTube Channels:
  - Subscribe to channels like 3Blue1Brown, Sentdex, and Siraj Raval for engaging explanations and tutorials on machine learning concepts.
8. Blogs:
  - Read blogs like KDnuggets, Towards Data Science, and Machine Learning Mastery for insightful articles on machine learning topics.
9. Online Communities:
  - Engage with researchers and practitioners through social media platforms like Twitter, LinkedIn, and GitHub.
10. Network:
  - Build connections with professionals in the field by attending meetups, joining professional organizations, and participating in hackathons.
11. Projects:
  - Apply your knowledge by working on personal projects or collaborating on group projects to reinforce understanding and build skills.

Remember, while these methods can help you gain knowledge quickly, consistent effort over time will lead to deeper understanding and better retention of information. It's essential to combine different approaches to get a well-rounded understanding of machine learning.


Llama 3.2 3B gen-z response to prompt 2: 

Here are some tips to help you study and gain knowledge about machine learning quickly:

- Start with the basics: Understand supervised and unsupervised learning, regression, classification, etc.
- Online courses: Use online resources like Coursera, edX, and Udemy to learn from experts.
- Books: Read books like "Pattern Classification" by Knight, "Machine Learning" by Bishop, and "Deep Learning" by Ng.
- Practice: Apply what you’ve learned by working on projects or kaggle competitions.
- Join communities: Participate in forums like Reddit’s r/MachineLearning and Stack Overflow to connect with others who share your interests.
- Watch videos: YouTube channels like 3BLOO, Machine Learning Mastery, and Sirhuman offer valuable insights.
- Hands-on experience: Try out different algorithms and tools to see how they work.
- Read research papers: Stay up-to-date with the latest advancements in the field by reading papers on arXiv and ResearchGate.

By following these steps, you can quickly gain knowledge about machine learning and start applying it in your work or projects.

### Conclusion

The results show that Llama 3B were slightly better at explaining the meaning of the senctence in prompt 1 in comparison to Llama 1B. However, it took approximately 40 seconds for Llama 1B to generate the response and it took approximately 300 seconds for Llama 3B FineTome and Llama 3B gen-z slang to generate its reponse. Prompt 1 was used to compare the finetuning with the gen-z dataset to the FineTome one. However, the genz slang model gave a worse response to the sentence in comparison to the other models. This might have been because of formatting mistakes with the gen-z dataset or that the dataset was small. 

The results for prompt 2 shows similar responses from all models. It took approximately 80 seconds for the 1B model and 300 for the 3B modell, regradless of the chosen fine tuning dataset.

Since we are performing inference on a CPU and the difference in responses between the Llama 3B and 1B models is minimal, we have chosen the Llama 3.2 1B model as our final selection, prioritizing faster response times for the user.