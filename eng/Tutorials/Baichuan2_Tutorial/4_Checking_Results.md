---
icon: terminal
tags: [tutorial, baichuan]
order: 40
---

# 4. Checking Training Results

Similar to the previous chapter, when you execute the **`train_baichuan2_13b.py`** script, the resulting model will be saved in the **`baichuan_code_generation`** directory. This model, stored as a pure PyTorch parameter file, is fully compatible not only with the MoAI Platform but also with regular GPU servers.

You can test the trained model using the **`inference_baichuan.py`** script located under the **`tutorial`** directory of the GitHub repository you downloaded earlier.

```python
# tutorial/inference_baichuan.py
...
input_text = f"[INST] I can no longer afford order 11234, cancel it [/INST]"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
generated_text_ids = input_ids.cuda()

with torch.no_grad():
    # Generate python function
    output = model.generate(generated_text_ids, max_new_tokens=MAX_NEW_TOKENS)

    # Decode generated tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Baichuan: {generated_text}")

...
```

Run the code below.

```bash
~/quickstart$ python tutorial/inference_baichuan.py
```

Upon inspecting the output, you can verify that the model has generated appropriate responses to the prompts.

```bash
### Input
input_text = f"[INST] I can no longer afford order 11234, cancel it [/INST]"

### Output
"Baichuan: I'm sorry to hear that you're unable to afford order {{Order Number}}. We understand the importance of keeping your expenses under control. To assist you with canceling your order, please follow these steps:

1. Log in to your account: Access our platform by signing in to your {{Online Company Portal Info}}.
2. Locate your order: Once logged in, navigate to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section.
3. Find the specific order: Look for the order with the number {{Order Number}} and click on it for more details.
4. Initiate the cancellation: Within the order details, you should find an option labeled '{{Online Order Interaction}}'. Please select this option.
5. Confirm the cancellation: The system may ask you to confirm the cancellation. Please provide any necessary details or feedback.

If you encounter any difficulties or have further questions, our dedicated team is available during {{Customer Support Hours}} at {{Customer Support Phone Number}} or through the Live Chat feature on our {{Website URL}}. We appreciate your understanding and look forward to resolving this matter for you.

Is there anything else we can assist you with? Your satisfaction is our top priority.

Thank you for reaching out!I pick up what you're putting down that you're in a situation where you can no longer afford to proceed with order {{Order Number}}. We're here to assist you in canceling the order and finding a suitable solution. Kindly follow the steps below: 1. Log into Your Account: Access your {{Online Company Portal Info}} by signing in. 2. Locate Your Order: Navigate to the '{{Online Order Interaction}}' or '{{Online Order Interaction}}' section. 3. Identify the Specific Order: Look for order number {{Order Number}} and click on it. 4. Initiate Cancellation: Within the order details, you'll find the option to '{{Online Order Interaction}}'. Please select it to begin the process. 5. Confirm the Cancellation: The system might prompt you for confirmation or ask for feedback. Please provide the necessary information to proceed. Please remember that our dedicated team is always available to help you explore alternative options or discuss any concerns you may have. Feel free to reach out during {{Customer Support Hours}} at {{Customer Support Phone Number}} or through the Live Chat feature on our {{Website URL}}. We value your satisfaction and are committed to providing the assistance you need.
```