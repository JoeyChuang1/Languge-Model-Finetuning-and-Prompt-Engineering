I employed a combination of fine-tuning and soft prompting techniques to tailor a DistilBERT model for natural language inference (NLI) tasks using NLI data. Here's a breakdown of these approaches:

Fine-tuning: This technique involves adjusting the internal parameters of a pre-trained model (like DistilBERT) on a specific task (NLI in this case). It essentially refines the model's existing knowledge to excel at the new task.

Soft prompting:  Unlike traditional prompting where explicit instructions are provided, soft prompting incorporates task-specific information directly into the model's input. This is achieved by strategically adding learnable parameters that guide the model towards the desired outcome. It offers greater flexibility compared to fixed prompts.

Normal prompting: This is the conventional method where clear instructions or cues are included at the beginning of the input data to guide the model towards the desired task.

By combining fine-tuning with soft prompting, I was able to effectively adapt DistilBERT for NLI tasks, achieving results comparable to those reported in the 'The Power of Scale for Parameter-Efficient Prompt Tuning' paper. This approach demonstrates the power of combining pre-trained models with strategic prompting techniques to achieve state-of-the-art performance while maintaining model efficiency.