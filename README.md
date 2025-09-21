# Fine-Tuning MedGemma on Brain Cancer MRI Data

This project demonstrates the process of fine-tuning `google/medgemma-4b-it`, a multimodal model, on a dataset of brain cancer MRI scans. The goal is to create a specialized vision-language assistant capable of identifying the type of brain tumor from an image and providing a detailed, empathetic explanation in response to a user's query.

## Project Overview

The notebook performs the following key steps:

1.  **Data Preparation**: Loads and preprocesses a multimodal dataset consisting of brain MRI images and corresponding textual diagnostic summaries.
2.  **Model Configuration**: Sets up the `MedGemma` model for efficient fine-tuning using 4-bit quantization and LoRA (Low-Rank Adaptation).
3.  **Fine-Tuning**: Uses the TRL (Transformer Reinforcement Learning) library's `SFTTrainer` to fine-tune the model on a structured conversational dataset.
4.  **Inference**: Shows how to load the fine-tuned model and use it to generate a diagnostic summary for a new brain scan image and question.

## Model and Dataset

  * **Base Model**: `google/medgemma-4b-it`. This is a state-of-the-art vision-language model from Google, specifically tuned for medical applications.
  * **Dataset**: The training data is a combination of two sources:
      * **Images**: [Brain Cancer MRI Dataset from Kaggle](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset), containing images classified into three categories: `brain_glioma`, `brain_menin`, and `brain_tumor`. 
      * **Text**: A custom-generated JSON file containing detailed explanations, next steps, and potential follow-up questions for each tumor type made by LLM!

   * Model available here: https://huggingface.co/Manishram/medgemma-brain-cancer-adapter

## Methodology

### 1\. Data Loading and Preprocessing

  - The image dataset is loaded from Kaggle, and a balanced subset is created with 91 images from each of the three classes (`glioma`, `meningioma`, `tumor`).
  - The textual data from the JSON file is loaded into a Pandas DataFrame.
  - The image and text datasets are merged to create a final, multimodal dataset where each image is paired with its corresponding diagnostic text and user query.

### 2\. Prompt Formatting

A function (`format_data`) is used to structure each data sample into a conversational format suitable for the `SFTTrainer`.

  * **User Prompt**:
    ```
    After looking at the image carefully What is the most likely type of brain cancer shown in the MRI image? 
    A:brain_glioma
    B:brain_menin
    C:brain_tumor. 
    I also have a follow up question {example['user_qs']}
    ```
  * **Assistant Response**:
    ```
    Most likely this image is a {BRAIN_CANCER_CLASSES[example['label']]}
    Here are some helpful details, along with thoughtful suggestions for what you might consider next:
    {example['assistant_answer']}
    ```

### 3\. Model Fine-Tuning

  - The `MedGemma` model is loaded using 4-bit quantization (`BitsAndBytesConfig`) to reduce memory requirements.
  - **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA** (`LoraConfig`) is applied to efficiently fine-tune the model by only updating a small subset of its parameters.
  - The `SFTTrainer` handles the training loop, data collation, and optimization.

## How to Use

1.  **Setup**: Install the required dependencies.
    ```bash
    pip install --upgrade --quiet transformers bitsandbytes datasets evaluate peft trl scikit-learn kaggle
    ```
2.  **Training**: Run the cells in the notebook sequentially to download the data, preprocess it, and execute `trainer.train()`.
3.  **Inference**: After training, the fine-tuned model adapters can be loaded and merged with the base model. The provided `chat_with_my_model` function can then be used to generate predictions on new images.

## Example Output

**User Input**:

  * **Image**: An MRI scan showing a brain glioma.
  * **Question**: "My brain scan report uses the words 'mass' and 'swelling.' Can you explain this to me in simple terms?"

**Model's Fine-Tuned Output**:

```
Most likely this image is a A:brain_glioma
Here are some helpful details, along with thoughtful suggestions for what you might consider next:
 ## Diagnostic Summary:

**Summary:**
The scan reveals a lesion in the brain consistent with a glioma.
The good news is that gliomas are a very treatable type of brain tumor, and there are many effective therapies available.

**Next Steps:**
* A discussion of the different treatment options, including surgery, radiation, chemotherapy, and clinical trials.
* A plan to get a tissue sample for diagnosis.
* Connecting with support services for you and your family.

**Common Follow-up Questions:**
* What is the most effective treatment for my type of glioma?
* What is the recovery process like after treatment?

** Final thoughts:**
This is a challenging diagnosis, but you are not alone. Your medical team will create a comprehensive plan to give you the best possible care and quality of life.
```
